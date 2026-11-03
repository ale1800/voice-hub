"""Web page (Home Assistant ingress panel): satellites, people, recent commands.

Only reachable through Home Assistant: when running as an add-on, requests that
don't come from the Supervisor ingress proxy are refused.
"""
from __future__ import annotations

import asyncio
import logging
import os
import wave

from aiohttp import web

from capture import Capture
from hub import Hub
from voiceprint import RATE
import ring
import ondevice

_LOG = logging.getLogger("voice-hub.web")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
INGRESS_PROXY = {"172.30.32.2", "127.0.0.1"}

# Semantic groups for a satellite's statistics (sensors only: the page shows,
# it doesn't control). Keywords are matched against the entity name without the
# device name (lower case, "_" = " "); first match wins, so the more specific
# groups come first. A few common non-English words are included because
# ESPHome configs are often written in the owner's language.
GROUPS = [
    ("voice", "Voice assistant", ["assist", "wake word", "finished speaking", "vad", "rilevamento vocale",
                                  "pipeline", "satellite"]),
    ("collect", "Recording & training data", ["capture", "collect", "record", "raccolta", "registra",
                                              "dataset", "negativ"]),
    ("intercom", "Intercom & calls", ["voip", "intercom", "sip", "call", "chiama", "rispondi", "riaggancia"]),
    ("audio", "Microphones & echo cancellation", ["aec", "erle", "echo_cancel", "reference", "mic", "gain",
                                                  "noise", "beam", "doa", "direction", "agc"]),
    ("speaker", "Speaker", ["volume", "mute", "speaker", "equalizer", "bass", "treble"]),
    ("light", "Lights", ["led", "light", "brightness"]),
    ("network", "Network", ["wifi", "wi fi", "signal", "rssi", "ssid", "bssid", "ip address", "mac address",
                            "connected", "ethernet"]),
    ("system", "System", ["heap", "psram", "memory", "loop", "uptime", "restart", "riavvia", "reboot",
                          "firmware", "version", "update.", "cpu", "temperature", "blocco", "reset", "safe mode"]),
]


DISPLAY_ORDER = ["audio", "network", "system", "voice", "speaker", "light", "intercom", "collect", "other"]


def entity_group(entity_id: str, name: str) -> str:
    text = name.lower().replace("_", " ")
    for key, _, words in GROUPS:
        if any(w in text for w in words):
            return key
    return "other"


class Web:
    def __init__(self, hub: Hub, capture: Capture | None) -> None:
        self.hub, self.capture = hub, capture
        self.ring = ring.RingStore(hub.args.data)
        self.ring_nodes: dict[str, str] = {}   # device_id -> board node name, from the last state read
        # 25 MB: an uploaded wake word head (aiohttp's default is 1 MB).
        self.app = web.Application(middlewares=[self._only_ingress], client_max_size=25 * 1024 * 1024)
        self.app.add_routes([
            web.get("/", self.index),
            web.get("/api/state", self.state),
            web.get("/api/audio/{name}", self.audio),
            web.post("/api/people", self.enroll),
            web.delete("/api/people/{name}", self.remove_person),
            web.post("/api/people/{name}/remove", self.remove_source),
            web.delete("/api/recordings/{name}", self.delete_recording),
            web.post("/api/enroll/start", self.guided_start),
            web.post("/api/enroll/goto", self.guided_goto),
            web.post("/api/enroll/skip", self.guided_skip),
            web.post("/api/enroll/cancel", self.guided_cancel),
            web.post("/api/enroll/save", self.guided_save),
            web.post("/api/wake/threshold", self.wake_threshold),
            web.get("/api/wake/clip/{name}", self.wake_clip),
            web.post("/api/agent", self.agent_save),
            web.post("/api/wake/models", self.wake_add_model),
            web.post("/api/wake/model", self.wake_set_model),
            web.delete("/api/wake/models/{name}", self.wake_remove_model),
            web.post("/api/ring/save", self.ring_save),
            web.post("/api/ring/preview", self.ring_preview),
            web.post("/api/ring/reset", self.ring_reset),
            web.post("/api/ring/stop", self.ring_stop),
            web.post("/api/ondevice/upload", self.ondevice_upload),
            web.post("/api/ondevice/remove", self.ondevice_remove),
        ])

    @web.middleware
    async def _only_ingress(self, request: web.Request, handler):
        if os.environ.get("SUPERVISOR_TOKEN") and request.remote not in INGRESS_PROXY:
            raise web.HTTPForbidden(text="Open Voice Hub from the Home Assistant sidebar.")
        return await handler(request)

    async def run(self, port: int) -> None:
        runner = web.AppRunner(self.app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", port).start()
        _LOG.info("web page on port %d", port)

    async def index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(os.path.join(STATIC, "index.html"),
                                headers={"Cache-Control": "no-store"})

    async def _ha_states(self) -> dict[str, dict]:
        try:
            async with self.hub.session.get(f"{self.hub.ha_url}/api/states", headers=self.hub.headers()) as r:
                return {s["entity_id"]: s for s in await r.json()}
        except Exception as e:
            _LOG.warning("could not read Home Assistant states: %s", e)
            return {}

    async def _ha_language(self) -> str:
        return await self.hub.language()

    async def state(self, request: web.Request) -> web.Response:
        states = await self._ha_states()
        devices = []
        if self.capture is not None:
            for sat in self.capture.sats.values():
                st = states.get(sat.entity_id, {})
                groups: dict[str, list] = {}
                for ent in self.capture.device_entities.get(sat.device_id, []):
                    eid = ent["entity_id"]
                    s = states.get(eid)
                    if s is None:
                        continue
                    a = s.get("attributes", {})
                    name = a.get("friendly_name", eid)
                    if name.lower().startswith(sat.name.lower() + " "):
                        name = name[len(sat.name) + 1:]
                    item = {"entity_id": eid, "name": name, "state": s["state"],
                            "unit": a.get("unit_of_measurement", ""), "category": ent["category"],
                            "device_class": a.get("device_class")}
                    if len(item["state"]) > 40:      # long text status (e.g. endpoint strings): not a statistic
                        continue
                    groups.setdefault(entity_group(eid, name), []).append(item)
                order = {None: 0, "config": 1, "diagnostic": 2}
                sensors = [{"key": k, "title": t,
                            "entities": sorted(groups[k], key=lambda x: (order.get(x["category"], 3), x["name"].lower()))}
                           for k, t, _ in sorted(GROUPS + [("other", "Other", [])],
                                                 key=lambda g: DISPLAY_ORDER.index(g[0]))
                           if k in groups]
                # VoIP Stack intercom: the device has a `..._voip_state` sensor (idle, calling,
                # ringing, in_call, ...). None = no intercom on this satellite, no call button.
                voip = next((states[e["entity_id"]]["state"] for e in self.capture.device_entities.get(sat.device_id, [])
                             if e["entity_id"].endswith("_voip_state") and e["entity_id"] in states), None)
                node = ring.node_of(self.capture.device_entities.get(sat.device_id, []), states)
                if node:
                    self.ring_nodes[sat.device_id] = node
                devices.append({
                    "device_id": sat.device_id, "name": sat.name, "entity_id": sat.entity_id,
                    "ring": ({"designs": self.ring.get(sat.device_id),
                              "stock": ring.stock_colour(self.capture.device_entities.get(sat.device_id, []), states)}
                             if node else None),
                    "ondevice": ondevice.status_of(self.capture.device_entities.get(sat.device_id, []), states),
                    "state": st.get("state", "unknown"),
                    "online": st.get("state", "unavailable") not in ("unavailable", "unknown"),
                    "voip_state": voip,
                    "pipeline": states.get(sat.pipeline_entity or "", {}).get("state"),
                    "last_command": self.capture.last_command.get(sat.device_id),
                    "groups": sensors,
                })
        reg = self.hub.registry
        reg.reload_if_changed()
        people = [{"name": n, "count": len(e), "sources": reg.sources.get(n, [])}
                  for n, e in sorted(reg.people.items())]
        a = self.hub.args
        return web.json_response({
            "language": await self._ha_language(),
            "devices": devices, "people": people, "recent": list(reversed(self.hub.recent)),
            "enrollment": (e.as_dict() if (e := self.hub.enrollment.active()) else None),
            "wake": self.hub.wake.summary() if self.hub.wake else None,
            "agent": ({"settings": self.hub.agent.settings, "last": self.hub.agent.last,
                       "agents": sorted(({"entity_id": e, "name": s.get("attributes", {}).get("friendly_name", e)}
                                         for e, s in states.items() if e.startswith("conversation.")),
                                        key=lambda a: a["name"].lower())}
                      if self.hub.agent else None),
            "settings": {"collect": self.hub.collect_all, "threshold": a.threshold,
                         "margin": a.margin, "min_duration": a.min_duration,
                         "capture": self.capture is not None, "wyoming": a.wyoming},
        })

    def _recording(self, name: str) -> str:
        if os.path.basename(name) != name or not name:
            raise web.HTTPNotFound()
        fn = os.path.join(self.hub.collection, name + ".wav")
        if not os.path.exists(fn):
            raise web.HTTPNotFound()
        return fn

    async def audio(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(self._recording(request.match_info["name"]),
                                headers={"Content-Type": "audio/wav"})

    @staticmethod
    def _name(body: dict) -> str:
        name = (body.get("name") or "").strip()
        if not name or len(name) > 40 or name.lower() == "unknown":
            raise web.HTTPBadRequest(text="Pick a name (not 'unknown', max 40 characters).")
        return name

    async def enroll(self, request: web.Request) -> web.Response:
        body = await request.json()
        name = self._name(body)
        files = body.get("files") or []
        if not files:
            raise web.HTTPBadRequest(text="Pick at least one recording.")
        return web.json_response(await self._enroll_files(name, files))

    async def _enroll_files(self, name: str, files: list[str]) -> dict:
        added, skipped = 0, []
        for f in files:
            fn = self._recording(f)
            with wave.open(fn) as w:
                pcm = w.readframes(w.getnframes())
            if len(pcm) / 2 / RATE < self.hub.args.min_duration:
                skipped.append(f)
                continue
            emb = await asyncio.get_running_loop().run_in_executor(None, self.hub.voiceprint.compute, pcm)
            self.hub.registry.add(name, emb, f)
            added += 1
        _LOG.info("enrolled %s: +%d recordings (%d too short)", name, added, len(skipped))
        return {"added": added, "skipped": skipped}

    # Guided enrollment (enrollment.py): the page polls /api/state for progress.
    async def guided_start(self, request: web.Request) -> web.Response:
        body = await request.json()
        name = self._name(body)
        device_id = body.get("device_id") or None
        device_name = "any satellite"
        if device_id and self.capture is not None:
            device_name = next((s.name for s in self.capture.sats.values() if s.device_id == device_id), device_id)
        self.hub.enrollment.start(name, device_id, device_name, await self.hub.language())
        _LOG.info("guided enrollment of %s started on %s", name, device_name)
        return web.json_response({"ok": True})

    async def guided_goto(self, request: web.Request) -> web.Response:
        self.hub.enrollment.goto(int((await request.json()).get("index", -1)))
        return web.json_response({"ok": True})

    async def guided_skip(self, request: web.Request) -> web.Response:
        self.hub.enrollment.skip()
        return web.json_response({"ok": True})

    async def guided_cancel(self, request: web.Request) -> web.Response:
        self.hub.enrollment.cancel()
        return web.json_response({"ok": True})

    async def guided_save(self, request: web.Request) -> web.Response:
        s = self.hub.enrollment.active()
        if s is None:
            raise web.HTTPNotFound(text="No enrollment in progress.")
        files = [st.file for st in s.steps if st.file]
        if not files:
            raise web.HTTPBadRequest(text="No samples recorded yet.")
        result = await self._enroll_files(s.name, files)
        self.hub.enrollment.cancel()
        return web.json_response(result)

    async def agent_save(self, request: web.Request) -> web.Response:
        if not self.hub.agent:
            raise web.HTTPNotFound(text="the conversation agent is off")
        return web.json_response(self.hub.agent.save(await request.json()))

    def _wake(self):
        if not self.hub.wake:
            raise web.HTTPNotFound(text="wake word engine is off")
        return self.hub.wake

    async def wake_add_model(self, request: web.Request) -> web.Response:
        wake = self._wake()
        form = await request.post()
        f = form.get("file")
        if f is None or not hasattr(f, "file"):
            raise web.HTTPBadRequest(text="No file sent.")
        blob = f.file.read()
        try:
            name = await asyncio.get_running_loop().run_in_executor(None, wake.add_model, f.filename, blob)
        except ValueError as e:
            raise web.HTTPBadRequest(text=str(e))
        return web.json_response({"added": name})

    async def wake_set_model(self, request: web.Request) -> web.Response:
        wake = self._wake()
        body = await request.json()
        try:
            await asyncio.get_running_loop().run_in_executor(None, wake.set_model, str(body["file"]))
        except ValueError as e:
            raise web.HTTPBadRequest(text=str(e))
        return web.json_response({"active": wake.head_file})

    async def wake_remove_model(self, request: web.Request) -> web.Response:
        wake = self._wake()
        try:
            await asyncio.get_running_loop().run_in_executor(None, wake.remove_model, request.match_info["name"])
        except ValueError as e:
            raise web.HTTPBadRequest(text=str(e))
        return web.json_response({"active": wake.head_file})

    async def wake_threshold(self, request: web.Request) -> web.Response:
        if not self.hub.wake:
            raise web.HTTPNotFound(text="wake word engine is off")
        body = await request.json()
        self.hub.wake.set_threshold(float(body["value"]))
        return web.json_response({"threshold": self.hub.wake.threshold})

    async def wake_clip(self, request: web.Request) -> web.FileResponse:
        name = os.path.basename(request.match_info["name"])
        path = os.path.join(self.hub.wake.rejected_dir, name) if self.hub.wake else ""
        if not name.endswith(".wav") or not os.path.exists(path):
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Content-Type": "audio/wav"})

    async def remove_person(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if not self.hub.registry.remove(name):
            raise web.HTTPNotFound()
        _LOG.info("removed %s", name)
        return web.json_response({"ok": True})

    async def remove_source(self, request: web.Request) -> web.Response:
        body = await request.json()
        if not self.hub.registry.remove_source(request.match_info["name"], body.get("file", "")):
            raise web.HTTPNotFound()
        return web.json_response({"ok": True})

    async def delete_recording(self, request: web.Request) -> web.Response:
        if not self.hub.delete_recording(request.match_info["name"]):
            raise web.HTTPNotFound()
        return web.json_response({"ok": True})

    # ═══ LED ring designs (LocalDot) ═══
    async def _ring_call(self, device_id: str, action: str, data: dict) -> None:
        node = self.ring_nodes.get(device_id)
        if not node:
            raise web.HTTPBadRequest(text="This satellite is not a LocalDot with this feature (update its firmware).")
        url = f"{self.hub.ha_url}/api/services/esphome/{node.replace('-', '_')}_{action}"
        async with self.hub.session.post(url, headers=self.hub.headers(), json=data) as r:
            if r.status >= 300:
                raise web.HTTPBadGateway(text=f"Home Assistant refused {action}: {r.status} {await r.text()}")

    # ═══ Custom on-device wake word (LocalDot) ═══
    async def _ondevice_status(self, device_id: str) -> str | None:
        states = await self._ha_states()
        return ondevice.status_of(self.capture.device_entities.get(device_id, []), states)

    async def _ondevice_wait(self, device_id: str, seconds: float = 25) -> str:
        """Waits for the board's sensor to report the result of ww_finish / ww_remove."""
        last = ""
        for _ in range(int(seconds * 2)):
            await asyncio.sleep(0.5)
            last = await self._ondevice_status(device_id) or ""
            if last.startswith(("active", "error", "none")):
                return last
        return last or "no answer from the board"

    async def ondevice_upload(self, request: web.Request) -> web.Response:
        form = await request.post()
        device_id = str(form.get("device_id", ""))
        model, manifest = form.get("model"), form.get("manifest")
        if not device_id or model is None or manifest is None:
            raise web.HTTPBadRequest(text="Pick the satellite, the .tflite file and the .json file.")
        data = model.file.read()
        try:
            args = ondevice.check(data, manifest.file.read())
        except ValueError as e:
            raise web.HTTPBadRequest(text=str(e))
        if await self._ondevice_status(device_id) is None:
            raise web.HTTPBadRequest(text="This satellite has no on-device wake word support (update its firmware).")
        start, chunks = ondevice.pieces(data)
        await self._ring_call(device_id, "ww_start", start)
        for c in chunks:
            await self._ring_call(device_id, "ww_chunk", c)
        await self._ring_call(device_id, "ww_finish", args)
        result = await self._ondevice_wait(device_id)
        _LOG.info("custom wake word «%s» to %s: %s", args["wake_word"], device_id, result)
        if not result.startswith("active"):
            raise web.HTTPBadGateway(text=f"The board did not accept the model: {result}")
        return web.json_response({"ok": True, "status": result})

    async def ondevice_remove(self, request: web.Request) -> web.Response:
        body = await request.json()
        await self._ring_call(body["device_id"], "ww_remove", {})
        result = await self._ondevice_wait(body["device_id"], 10)
        return web.json_response({"ok": True, "status": result})

    @staticmethod
    def _ring_body(body: dict) -> tuple[str, int, str, int]:
        occ = body.get("occasion")
        eff = body.get("effect", "solid")
        if occ not in ring.OCCASIONS or eff not in ring.EFFECTS:
            raise web.HTTPBadRequest(text="unknown occasion or effect")
        try:
            hexes = ring.colors_hex(body.get("colors") or [])
        except ValueError as e:
            raise web.HTTPBadRequest(text=str(e))
        # effect index in bits 0-1, speed code in bits 2-7 (firmware: localdot_ring_events.yaml)
        return occ, ring.OCCASIONS.index(occ), hexes, ring.EFFECTS.index(eff) + 4 * ring.speed_code(body.get("speed"))

    async def ring_save(self, request: web.Request) -> web.Response:
        body = await request.json()
        occ, i, hexes, eff = self._ring_body(body)
        await self._ring_call(body["device_id"], "anello_imposta", {"evento": i, "colori": hexes, "effetto": eff})
        self.ring.put(body["device_id"], occ, body["colors"], body.get("effect", "solid"), float(body.get("speed") or 1.0))
        return web.json_response({"ok": True})

    async def ring_preview(self, request: web.Request) -> web.Response:
        body = await request.json()
        _, _, hexes, eff = self._ring_body(body)
        # Live preview: the page sends every change and renews it while it is open (30 s each).
        secs = int(body.get("seconds", 4))
        await self._ring_call(body["device_id"], "anello_mostra", {"colori": hexes, "effetto": eff, "secondi": max(1, min(60, secs))})
        return web.json_response({"ok": True})

    async def ring_stop(self, request: web.Request) -> web.Response:
        body = await request.json()
        await self._ring_call(body["device_id"], "anello_ferma", {})
        return web.json_response({"ok": True})

    async def ring_reset(self, request: web.Request) -> web.Response:
        body = await request.json()
        occ = body.get("occasion")
        if occ not in ring.OCCASIONS:
            raise web.HTTPBadRequest(text="unknown occasion")
        await self._ring_call(body["device_id"], "anello_cancella", {"evento": ring.OCCASIONS.index(occ)})
        self.ring.remove(body["device_id"], occ)
        return web.json_response({"ok": True})


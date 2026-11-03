"""Listen to commands INSIDE Home Assistant, without changing any pipeline.

Every satellite already sends its command audio to Home Assistant, which
forwards it to the speech-to-text engine of the pipeline (Home Assistant Cloud,
Whisper, ...). Home Assistant has an official API to get a copy of that audio
for a device while it is being transcribed: `assist_pipeline/device/capture`
(the one used by the voice setup wizard). This module:

  1. watches the `assist_satellite.*` entities; when one turns `listening`
     (start of speech-to-text) it subscribes to that device's audio;
  2. when the audio stops (Home Assistant detected the end of the command), it
     computes the voiceprint and publishes who spoke, while speech-to-text is
     still running in the cloud;
  3. measures whether the result got there before the satellite turned
     `processing` (start of intent handling = before sentence automations run);
  4. after the run, reads the transcript from the pipeline debug history so the
     collected command is labelled with its text.

Only one capture per device can run at a time in Home Assistant: if you open
the voice debug tool on the same device, it takes over for that command.
Each capture adds a "started recording" line to the logbook, once per command.
"""
from __future__ import annotations

import asyncio
import base64
import itertools
import logging
import time
from dataclasses import dataclass, field

import aiohttp

from hub import Hub
from voiceprint import RATE

_LOG = logging.getLogger("voice-hub.capture")

GAP_S = 0.25           # no audio for this long = Home Assistant closed the stream
# Speed: the voiceprint (~1 s on a Home Assistant Yellow) is computed WHILE the
# person speaks, again every STEP_S of new audio. When the stream closes, a
# voiceprint covering all but the last LAG_OK_S (mostly the trailing silence
# Home Assistant waits for) is used at once instead of starting from scratch.
STEP_S = 0.5
LAG_OK_S = 1.0
CAPTURE_TIMEOUT_S = 30.0
REDISCOVER_S = 300.0


class HaWebSocket:
    """Minimal Home Assistant websocket client: calls and subscriptions."""

    def __init__(self, hub: Hub) -> None:
        self.hub = hub
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._subs: dict[int, callable] = {}

    async def connect(self) -> None:
        # Supervisor proxy: /core/websocket; Home Assistant directly: /api/websocket.
        path = "/websocket" if self.hub.ha_url == "http://supervisor/core" else "/api/websocket"
        url = self.hub.ha_url.replace("http", "ws", 1) + path
        self.ws = await self.hub.session.ws_connect(url, max_msg_size=0, heartbeat=30)
        await self.ws.receive_json()
        await self.ws.send_json({"type": "auth", "access_token": self.hub.ha_token})
        msg = await self.ws.receive_json()
        if msg.get("type") != "auth_ok":
            raise RuntimeError(f"Home Assistant websocket auth failed: {msg}")

    async def run(self) -> None:
        """Dispatch incoming messages until the connection closes."""
        async for m in self.ws:
            if m.type != aiohttp.WSMsgType.TEXT:
                break
            msg = m.json()
            mid = msg.get("id")
            if msg.get("type") == "result" and mid in self._pending:
                fut = self._pending.pop(mid)
                if not fut.done():
                    if msg.get("success"):
                        fut.set_result(msg.get("result"))
                    else:
                        fut.set_exception(RuntimeError(msg.get("error")))
            elif msg.get("type") == "event" and mid in self._subs:
                self._subs[mid](msg["event"])
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("websocket closed"))
        self._pending.clear()
        self._subs.clear()

    async def call(self, msg: dict, on_event=None):
        mid = next(self._ids)
        fut = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        if on_event is not None:
            self._subs[mid] = on_event
        await self.ws.send_json({**msg, "id": mid})
        try:
            return mid, await asyncio.wait_for(fut, 10)
        except Exception:
            self._subs.pop(mid, None)
            raise

    async def unsubscribe(self, mid: int) -> None:
        if self._subs.pop(mid, None) is not None:
            try:
                await self.call({"type": "unsubscribe_events", "subscription": mid})
            except Exception:
                pass


@dataclass
class Satellite:
    device_id: str
    name: str
    entity_id: str                    # assist_satellite.*
    pipeline_entity: str | None       # select.*_assistant


@dataclass
class Command:
    sat: Satellite
    started: float = field(default_factory=time.monotonic)
    audio: bytearray = field(default_factory=bytearray)
    last_chunk: float = 0.0
    sub_id: int | None = None
    finalized: asyncio.Event = field(default_factory=asyncio.Event)
    published_at: float | None = None
    processing_at: float | None = None
    base: str | None = None           # collected file, without extension
    person: str = "unknown"
    entry: dict | None = None         # row in hub.recent
    partial_emb: object = None        # voiceprint of the first partial_s seconds
    partial_s: float = 0.0
    partial_task: asyncio.Task | None = None
    text_task: asyncio.Task | None = None


class Capture:
    def __init__(self, hub: Hub) -> None:
        self.hub = hub
        self.ha = HaWebSocket(hub)
        self.sats: dict[str, Satellite] = {}          # by entity_id
        self.active: dict[str, Command] = {}          # by device_id
        self.last_command: dict[str, dict] = {}       # by device_id, for the web page
        self.device_entities: dict[str, list[dict]] = {}  # read-only sensors of each satellite device
        self._trigger_sub: int | None = None
        self._idle: dict[str, list[asyncio.Future]] = {}  # device_id -> waiters for `idle`
        self._done_at: dict[str, float] = {}           # device_id -> when its last command was published

    # ── for the conversation agent (agent.py) ──
    def satellite_of(self, device_id: str) -> Satellite | None:
        return next((s for s in self.sats.values() if s.device_id == device_id), None)

    async def speaker_now(self, device_id: str, timeout: float) -> str | None:
        """Who spoke this device's command that is being handled now.
        The transcript is out, so the audio is over: finish at once instead of
        waiting for the stream gap, and wait at most `timeout`."""
        cmd = self.active.get(device_id)
        if cmd is None:                         # already finished and published
            if time.monotonic() - self._done_at.get(device_id, -1e9) < 15:
                return (self.last_command.get(device_id) or {}).get("person")
            return None
        if not cmd.finalized.is_set():
            asyncio.create_task(self._finalize(cmd))
        end = time.monotonic() + timeout
        while cmd.published_at is None and time.monotonic() < end:
            await asyncio.sleep(0.02)
        return cmd.person if cmd.published_at is not None else None

    async def wait_idle(self, device_id: str, timeout: float) -> bool:
        fut = asyncio.get_running_loop().create_future()
        self._idle.setdefault(device_id, []).append(fut)
        try:
            await asyncio.wait_for(fut, timeout)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            if fut in self._idle.get(device_id, []):
                self._idle[device_id].remove(fut)

    async def run_forever(self) -> None:
        while True:
            try:
                await self.ha.connect()
                reader = asyncio.create_task(self.ha.run())
                await self._discover()
                watcher = asyncio.create_task(self._rediscover_loop())
                await reader
                watcher.cancel()
                _LOG.warning("Home Assistant websocket closed, reconnecting")
            except Exception as e:
                _LOG.warning("Home Assistant websocket: %s", e)
            self.active.clear()
            await asyncio.sleep(5)

    async def _rediscover_loop(self) -> None:
        while True:
            await asyncio.sleep(REDISCOVER_S)
            await self._discover()

    async def _discover(self) -> None:
        _, devices = await self.ha.call({"type": "assist_pipeline/device/list"})
        _, ents = await self.ha.call({"type": "config/entity_registry/list"})
        _, devreg = await self.ha.call({"type": "config/device_registry/list"})
        names = {d["id"]: d.get("name_by_user") or d.get("name") or d["id"] for d in devreg}
        sat_entity = {e["device_id"]: e["entity_id"] for e in ents
                      if e["entity_id"].startswith("assist_satellite.") and not e.get("disabled_by")}
        self.device_entities = {}
        for e in ents:
            if e.get("device_id") and not e.get("disabled_by") and not e.get("hidden_by") and \
                    e["entity_id"].split(".")[0] in ("sensor", "binary_sensor"):
                self.device_entities.setdefault(e["device_id"], []).append(
                    {"entity_id": e["entity_id"], "category": e.get("entity_category")})
        sats = {}
        for d in devices:
            ent = sat_entity.get(d["device_id"])
            if ent:
                sats[ent] = Satellite(d["device_id"], names.get(d["device_id"], d["device_id"]),
                                      ent, d.get("pipeline_entity"))
        if sats.keys() == self.sats.keys() and self._trigger_sub is not None:
            self.sats = sats
            return
        self.sats = sats
        if self._trigger_sub is not None:
            await self.ha.unsubscribe(self._trigger_sub)
        if not sats:
            _LOG.warning("no Assist satellites found")
            self._trigger_sub = None
            return
        self._trigger_sub, _ = await self.ha.call(
            {"type": "subscribe_trigger",
             "trigger": {"platform": "state", "entity_id": sorted(sats)}},
            on_event=self._on_state)
        _LOG.info("watching %s", ", ".join(f"{s.name} ({s.entity_id})" for s in sats.values()))

    def _on_state(self, event: dict) -> None:
        trig = event.get("variables", {}).get("trigger", {})
        to = (trig.get("to_state") or {}).get("state")
        sat = self.sats.get(trig.get("entity_id"))
        if sat is None:
            return
        if to == "idle":
            for fut in self._idle.pop(sat.device_id, []):
                if not fut.done():
                    fut.set_result(True)
        if to == "listening":
            asyncio.create_task(self._start(sat))
        elif to == "processing" and (cmd := self.active.get(sat.device_id)):
            cmd.processing_at = time.monotonic()
            # Speech-to-text is done: read the transcript now, not after the reply.
            cmd.text_task = asyncio.create_task(self._label(cmd))

    async def _start(self, sat: Satellite) -> None:
        old = self.active.get(sat.device_id)
        if old is not None and old.sub_id is not None:
            await self.ha.unsubscribe(old.sub_id)
        cmd = Command(sat)
        self.active[sat.device_id] = cmd

        def on_audio(ev: dict) -> None:
            if ev.get("type") == "audio":
                cmd.audio += base64.b64decode(ev["audio"])
                cmd.last_chunk = time.monotonic()
            elif ev.get("type") == "end":
                asyncio.create_task(self._end(cmd))

        try:
            cmd.sub_id, _ = await self.ha.call(
                {"type": "assist_pipeline/device/capture", "device_id": sat.device_id,
                 "timeout": CAPTURE_TIMEOUT_S}, on_event=on_audio)
        except Exception as e:
            _LOG.warning("%s: capture failed: %s", sat.name, e)
            self.active.pop(sat.device_id, None)
            return
        # Watch the stream: voiceprint as it grows, finalize when it stops.
        while not cmd.finalized.is_set():
            await asyncio.sleep(0.05)
            if cmd.audio and time.monotonic() - cmd.last_chunk >= GAP_S:
                await self._finalize(cmd)
                break
            dur = len(cmd.audio) / 2 / RATE
            if (cmd.partial_task is None or cmd.partial_task.done()) and \
                    dur >= self.hub.args.min_duration and dur - cmd.partial_s >= STEP_S:
                cmd.partial_task = asyncio.create_task(self._partial(cmd, bytes(cmd.audio)))

    async def _partial(self, cmd: Command, pcm: bytes) -> None:
        emb = await self.hub.embed(pcm)
        if len(pcm) / 2 / RATE > cmd.partial_s:
            cmd.partial_emb, cmd.partial_s = emb, len(pcm) / 2 / RATE

    async def _finalize(self, cmd: Command) -> None:
        if cmd.finalized.is_set():
            return
        cmd.finalized.set()
        pcm = bytes(cmd.audio)
        t0 = time.monotonic()
        duration = len(pcm) / 2 / RATE
        if duration - cmd.partial_s > LAG_OK_S and cmd.partial_task is not None and not cmd.partial_task.done():
            await cmd.partial_task            # nearly there: cheaper than starting over
        if cmd.partial_emb is not None and duration - cmd.partial_s <= LAG_OK_S:
            result, early = self.hub.decide(cmd.partial_emb, duration, cmd.sat.device_id), cmd.partial_s
        else:
            result, _ = await self.hub.identify(pcm, cmd.sat.device_id)
            early = duration
        t_vp = time.monotonic() - t0
        cmd.person = result.person
        await self.hub.publish(result, device_id=cmd.sat.device_id, device_name=cmd.sat.name)
        cmd.published_at = time.monotonic()
        self._done_at[cmd.sat.device_id] = cmd.published_at
        enrolling = self.hub.enrollment.wants(cmd.sat.device_id)
        cmd.base = self.hub.collect(pcm, result, force=enrolling, device_id=cmd.sat.device_id,
                                    device_name=cmd.sat.name)
        cmd.entry = self.hub.add_recent(result, cmd.base, device_id=cmd.sat.device_id,
                                        device_name=cmd.sat.name, voiceprint_ms=round(t_vp * 1000))
        if enrolling:
            self.hub.enrollment.offer(cmd.sat.device_id, cmd.entry["file"], cmd.entry, result.duration_s)
        self.last_command[cmd.sat.device_id] = cmd.entry
        _LOG.info("%s: %.1f s (voiceprint on %.1f s) | %s (%.2f, runner-up %.2f) %s | ready %.0f ms after the stream",
                  cmd.sat.name, result.duration_s, early, result.person, result.score, result.runner_up,
                  result.reason, t_vp * 1000)

    async def _end(self, cmd: Command) -> None:
        """The pipeline run is over (or the capture timed out)."""
        await self._finalize(cmd)
        while cmd.published_at is None:         # another task is still identifying
            await asyncio.sleep(0.02)
        if self.active.get(cmd.sat.device_id) is cmd:
            self.active.pop(cmd.sat.device_id)
        cmd.sub_id = None
        if cmd.processing_at is None:
            race = "no intent stage"
        else:
            lead = (cmd.processing_at - cmd.published_at) * 1000
            if lead >= 0:
                race = f"published {lead:.0f} ms BEFORE intent"
            elif self.hub.agent is not None and any(t.get("device") == cmd.sat.name and time.time() - t["time"] < 30
                                                    for t in self.hub.agent.last[:3]):
                # Voice Hub is the agent: the intent stage began by waiting for this.
                race = f"published {-lead:.0f} ms into the intent stage, before the Voice Hub agent passed the command on"
            else:
                race = f"published {-lead:.0f} ms AFTER intent (too late for sentence automations)"
        if cmd.text_task is None:
            cmd.text_task = asyncio.create_task(self._label(cmd))
        text = await cmd.text_task
        _LOG.info("%s: \"%s\" -> %s | %s", cmd.sat.name, text, cmd.person, race)

    async def _label(self, cmd: Command) -> str | None:
        """Attach the transcript and the race result to the command."""
        await cmd.finalized.wait()
        while cmd.published_at is None:         # _finalize still identifying
            await asyncio.sleep(0.02)
        text = await self._transcript(cmd)
        fields = {"text": text, "race_ms": None if cmd.processing_at is None
                  else round((cmd.processing_at - cmd.published_at) * 1000)}
        self.hub.update_collected(cmd.base, fields)
        if cmd.entry is not None:
            cmd.entry.update(fields)
            self.hub.enrollment.confirm(cmd.entry.get("file"), text)
        return text

    async def _transcript(self, cmd: Command) -> str | None:
        """Read the speech-to-text output of this run from the pipeline debug history."""
        try:
            _, pl = await self.ha.call({"type": "assist_pipeline/pipeline/list"})
            pipelines = pl["pipelines"]
            wanted = None
            if cmd.sat.pipeline_entity:
                _, states = await self.ha.call({"type": "get_states"})
                st = next((s["state"] for s in states if s["entity_id"] == cmd.sat.pipeline_entity), None)
                wanted = next((p["id"] for p in pipelines if p["name"] == st), None)
                if wanted is None:
                    wanted = pl.get("preferred_pipeline")
            ids = [wanted] if wanted else [p["id"] for p in pipelines]
            for pid in ids:
                _, runs = await self.ha.call({"type": "assist_pipeline/pipeline_debug/list", "pipeline_id": pid})
                for run in sorted(runs["pipeline_runs"], key=lambda r: r["timestamp"], reverse=True)[:3]:
                    _, dbg = await self.ha.call({"type": "assist_pipeline/pipeline_debug/get",
                                                 "pipeline_id": pid, "pipeline_run_id": run["pipeline_run_id"]})
                    text, dev = None, None
                    for ev in dbg["events"]:
                        if ev["type"] == "stt-end":
                            text = (ev.get("data") or {}).get("stt_output", {}).get("text")
                        elif ev["type"] == "intent-start":
                            dev = (ev.get("data") or {}).get("device_id")
                    if dev == cmd.sat.device_id:
                        return text
        except Exception as e:
            _LOG.debug("transcript lookup failed: %s", e)
        return None

    async def satellite_at(self, started: float) -> str | None:
        """Which satellite's pipeline run began within a few seconds of `started`
        (a wake word session of wake.py: Home Assistant does not say who it is)."""
        from datetime import datetime
        try:
            _, pl = await self.ha.call({"type": "assist_pipeline/pipeline/list"})
            for p in pl["pipelines"]:
                _, runs = await self.ha.call({"type": "assist_pipeline/pipeline_debug/list", "pipeline_id": p["id"]})
                for run in sorted(runs["pipeline_runs"], key=lambda r: r["timestamp"], reverse=True)[:5]:
                    if abs(datetime.fromisoformat(run["timestamp"]).timestamp() - started) > 4:
                        continue
                    _, dbg = await self.ha.call({"type": "assist_pipeline/pipeline_debug/get",
                                                 "pipeline_id": p["id"], "pipeline_run_id": run["pipeline_run_id"]})
                    for ev in dbg["events"]:
                        sid = (ev.get("data") or {}).get("satellite_id")
                        if sid:
                            sat = self.sats.get(sid)
                            return sat.name if sat else sid
        except Exception as e:
            _LOG.debug("satellite lookup failed: %s", e)
        return None

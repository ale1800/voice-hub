"""Shared core: identify the speaker of a command, tell Home Assistant, keep
the audio for later enrollment.

Used by both audio sources:
  * capture.py - listens to the Assist pipeline inside Home Assistant (no
    change to your pipelines; the default);
  * server.py  - optional Wyoming speech-to-text pass-through.
"""
from __future__ import annotations

import asyncio
import glob
import io
import json
import logging
import os
import time
import wave
from collections import deque

import aiohttp

from enrollment import Enrollment
from voiceprint import RATE, Registry, Result, Settings, Voiceprint

_LOG = logging.getLogger("voice-hub")


def wav_from_pcm(pcm: bytes) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(pcm)
    return b.getvalue()


class Hub:
    def __init__(self, args) -> None:
        self.args = args
        self.voiceprint = Voiceprint(args.model, threads=args.threads)
        model = os.path.splitext(os.path.basename(args.model))[0]
        self.registry = Registry(args.data, Settings(args.min_duration, args.threshold, args.margin), model=model)
        if os.environ.get("SUPERVISOR_TOKEN"):
            self.ha_url, self.ha_token = "http://supervisor/core", os.environ["SUPERVISOR_TOKEN"]
        else:
            self.ha_url, self.ha_token = os.environ["HA_URL"].rstrip("/"), os.environ["HA_TOKEN"]
        # Recordings live here. With `collect` every command is saved; otherwise
        # only the ones taken by a guided enrollment.
        self.collection = os.path.join(args.data, "collection")
        self.collect_all = bool(args.collect)
        os.makedirs(self.collection, exist_ok=True)
        self.enrollment = Enrollment(args.language, args.min_duration)
        self.session: aiohttp.ClientSession | None = None
        self.wake = None                      # wake.WakeEngine with --wake
        self.agent = None                     # agent.Agent with --agent
        # Last commands for the web page (newest last). Each entry is a dict that
        # capture/server keep updating (transcript, race) after it is added.
        self.recent: deque[dict] = deque(maxlen=200)
        self._load_recent()
        self.reembed()

    def reembed(self) -> None:
        """Voiceprints from another model are not comparable: recompute every
        enrolled sample from its saved recording (a few seconds). A sample whose
        recording is gone is dropped, and said so."""
        reg = self.registry
        if not reg.stale:
            return
        _LOG.info("voiceprints made by %s, model is now %s: recomputing them", reg.file_model, reg.model)
        for name in list(reg.people):
            emb, src = [], []
            for f in reg.sources.get(name, []):
                try:
                    with wave.open(os.path.join(self.collection, f + ".wav")) as w:
                        pcm = w.readframes(w.getnframes())
                except (OSError, wave.Error):
                    _LOG.warning("%s: recording %s is missing, sample dropped", name, f or "?")
                    continue
                emb.append([float(v) for v in self.voiceprint.compute(pcm)])
                src.append(f)
            if emb:
                reg.people[name], reg.sources[name] = emb, src
            else:
                _LOG.warning("%s: no recording left, enroll again", name)
                reg.people.pop(name, None); reg.sources.pop(name, None)
        reg.save()
        _LOG.info("voiceprints recomputed: %s", ", ".join(f"{n} {len(e)}" for n, e in reg.people.items()))

    async def language(self) -> str:
        """The language to use: the add-on option if set, otherwise Home Assistant's own
        (read once from /api/config), otherwise English."""
        if self.args.language:
            return self.args.language
        if getattr(self, "_ha_language", None) is None:
            try:
                async with self.session.get(f"{self.ha_url}/api/config", headers=self.headers()) as r:
                    self._ha_language = (await r.json()).get("language") or "en"
            except Exception as e:
                _LOG.warning("could not read Home Assistant's language: %s", e)
                return "en"
        return self._ha_language

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.ha_token}"}

    async def embed(self, pcm: bytes):
        return await asyncio.get_running_loop().run_in_executor(None, self.voiceprint.compute, pcm)

    def decide(self, emb, duration: float, device_id: str | None = None) -> Result:
        result = self.registry.identify(emb, duration, device_id)
        if self.registry.settings.use_cohort and duration >= self.args.min_duration:
            self.registry.cohort.add(emb)   # numbers only: the signature to subtract
        return result

    async def identify(self, pcm: bytes, device_id: str | None = None) -> tuple[Result, float]:
        """Returns the result and the time spent on the voiceprint (s)."""
        duration = len(pcm) / 2 / RATE
        if duration <= 0.3:
            return Result("unknown", 0.0, 0.0, duration, "no audio"), 0.0
        t0 = time.monotonic()
        return self.decide(await self.embed(pcm), duration, device_id), time.monotonic() - t0

    def seed_cohort(self) -> None:
        """First start with saved commands but no cohort yet: build it from them
        (about one second per command, in a background thread)."""
        if not self.registry.settings.use_cohort or len(self.registry.cohort.items) >= self.registry.cohort.MIN:
            return
        n = 0
        for fn in sorted(glob.glob(os.path.join(self.collection, "*.wav"))):
            try:
                with wave.open(fn) as w:
                    pcm = w.readframes(w.getnframes())
            except (OSError, wave.Error):
                continue
            if len(pcm) / 2 / RATE < self.args.min_duration:
                continue
            self.registry.cohort.add(self.voiceprint.compute(pcm))
            n += 1
        _LOG.info("cohort seeded from %d saved commands (%d in total)", n, len(self.registry.cohort.items))

    async def publish(self, r: Result, *, device_id: str | None = None, device_name: str | None = None,
                      text: str | None = None) -> None:
        """Event `voice_hub_speaker` + state of `sensor.voice_hub_last_speaker`."""
        data = {"person": r.person, "score": round(r.score, 3), "runner_up": round(r.runner_up, 3),
                "duration_s": round(r.duration_s, 2), "reason": r.reason,
                "device_id": device_id, "device_name": device_name}
        if text is not None:
            data["text"] = text
        try:
            await self.session.post(f"{self.ha_url}/api/events/voice_hub_speaker", json=data,
                                    headers=self.headers())
            await self.session.post(f"{self.ha_url}/api/states/sensor.voice_hub_last_speaker",
                                    json={"state": r.person,
                                          "attributes": {**data, "friendly_name": "Last speaker",
                                                         "icon": "mdi:account-voice"}},
                                    headers=self.headers())
        except Exception as e:  # Home Assistant unreachable: the command still works
            _LOG.warning("could not publish to Home Assistant: %s", e)

    def _load_recent(self) -> None:
        for fn in sorted(glob.glob(os.path.join(self.collection, "*.json")))[-self.recent.maxlen:]:
            try:
                with open(fn) as f:
                    d = json.load(f)
            except (OSError, ValueError):
                continue
            name = os.path.basename(fn)[:-5]
            d.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S", time.strptime(name[:15], "%Y%m%d-%H%M%S")))
            d["file"] = name if os.path.exists(fn[:-5] + ".wav") else None
            self.recent.append(d)

    def add_recent(self, r: Result, base: str | None, **extra) -> dict:
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "person": r.person, "score": r.score,
                 "runner_up": r.runner_up, "duration_s": r.duration_s, "reason": r.reason,
                 "file": os.path.basename(base) if base else None, **extra}
        self.recent.append(entry)
        return entry

    def delete_recording(self, name: str) -> bool:
        if os.path.basename(name) != name or not name:
            return False
        found = False
        for ext in (".wav", ".json"):
            fn = os.path.join(self.collection, name + ext)
            if os.path.exists(fn):
                os.remove(fn)
                found = True
        for e in self.recent:
            if e.get("file") == name:
                e["file"] = None
        return found

    def collect(self, pcm: bytes, r: Result, force: bool = False, **extra) -> str | None:
        """Save the command as <collection>/<timestamp>.wav + .json. Returns the base path."""
        if not (self.collect_all or force) or not pcm:
            return None
        base = os.path.join(self.collection, time.strftime("%Y%m%d-%H%M%S"))
        n = 1
        while os.path.exists(base + ".wav"):   # two satellites in the same second
            n += 1
            base = os.path.join(self.collection, time.strftime("%Y%m%d-%H%M%S") + f"-{n}")
        with open(base + ".wav", "wb") as f:
            f.write(wav_from_pcm(pcm))
        self.update_collected(base, {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "person": r.person, "score": r.score, "scores": r.scores,
                                     "duration_s": r.duration_s, "reason": r.reason, **extra})
        return base

    @staticmethod
    def update_collected(base: str | None, fields: dict) -> None:
        if not base:
            return
        fn = base + ".json"
        try:
            with open(fn) as f:
                d = json.load(f)
        except FileNotFoundError:
            d = {}
        d.update(fields)
        with open(fn, "w") as f:
            json.dump(d, f, indent=1)

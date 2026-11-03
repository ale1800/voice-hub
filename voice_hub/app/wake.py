"""Wake word service (Wyoming) for Home Assistant: the second stage.

A LocalDot that already heard its wake word on the device can ask Home
Assistant to check it again ("second stage"); Home Assistant then streams the
dot's audio (including the ~2 s before the wake) to a wake word engine. This is
that engine, inside Voice Hub, so each check can be counted per satellite and
the rejected ones kept for listening.

Heads: the built-in ones next to the feature models, plus any uploaded from
the page (`<data>/wake_models`, openWakeWord `.onnx` with input (1, 16, 96)).
The one in use can be switched live; Home Assistant keeps seeing the same
wake word name, so its pipelines need no change.

Features: openWakeWord's own melspectrogram + speech embedding (oww_features),
then a model head over the last 16 embeddings (1.28 s) - by default the public
openWakeWord "hey_jarvis" head; any head with the same input (1, 16, 96) can be
dropped in (`--wake-model`).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import wave
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram
from wyoming.server import AsyncEventHandler
from wyoming.wake import Detect, Detection, NotDetected

from oww_features import OwwFeatures, OwwStream

_LOG = logging.getLogger("voice-hub.wake")
RATE = 16000
HOP = 1280                 # score every 80 ms, as openWakeWord
KEEP_S = 8                 # audio kept per session for a rejected clip
FEATURE_MODELS = {"melspectrogram.onnx", "embedding_model.onnx"}
MAX_HEAD_BYTES = 20 * 1024 * 1024


@dataclass
class Session:
    started: float                      # time.time()
    model: str
    scores: list[float] = field(default_factory=list)
    detected_at_ms: int | None = None
    satellite: str | None = None
    clip: str | None = None

    def as_dict(self) -> dict:
        return {"time": self.started, "model": self.model,
                "max_score": round(max(self.scores), 3) if self.scores else 0.0,
                "detected": self.detected_at_ms is not None, "detected_at_ms": self.detected_at_ms,
                "satellite": self.satellite, "clip": self.clip}


class WakeEngine:
    """Shared by every connection: the ONNX sessions and the statistics."""

    def __init__(self, model_dir: str, head: str, threshold: float, data: str,
                 save_rejected: bool, threads: int = 1) -> None:
        self.features = OwwFeatures(os.path.join(model_dir, "melspectrogram.onnx"),
                                    os.path.join(model_dir, "embedding_model.onnx"), threads)
        self.threads = threads
        self.builtin_dir = model_dir
        self.custom_dir = os.path.join(data, "wake_models")
        os.makedirs(self.custom_dir, exist_ok=True)
        default = head if os.path.isabs(head) else os.path.join(model_dir, head)
        # The name Home Assistant knows; it stays the same whichever head is in use.
        self.model_name = os.path.splitext(os.path.basename(default))[0]
        self.default_head = os.path.basename(default)
        self.settings_path = os.path.join(data, "wake.json")
        self.threshold = threshold
        saved: dict = {}
        try:
            saved = json.load(open(self.settings_path))
            self.threshold = float(saved["threshold"])
        except Exception:
            pass
        self.head = self._open_head(default)          # (session, input name), swapped as one
        self.stream_start = OwwStream.start(self.features)
        self.head_file = os.path.basename(default)
        if saved.get("model") and saved["model"] != self.head_file:
            try:
                self.set_model(saved["model"], persist=False)
            except Exception as e:
                _LOG.warning("saved wake model %s not usable (%s): back to %s", saved["model"], e, self.head_file)
        self.save_rejected = save_rejected
        self.rejected_dir = os.path.join(data, "wake_rejected")
        os.makedirs(self.rejected_dir, exist_ok=True)
        self.sessions: deque[Session] = deque(maxlen=500)

    def _save(self) -> None:
        with open(self.settings_path, "w") as f:
            json.dump({"threshold": self.threshold, "model": self.head_file}, f)

    def set_threshold(self, value: float) -> None:
        self.threshold = min(max(float(value), 0.01), 0.99)
        self._save()
        _LOG.info("wake threshold now %.2f", self.threshold)

    # ── heads ──
    def _open_head(self, path: str):
        """Load a head and check it takes 16 embeddings of 96 and gives one score."""
        o = ort.SessionOptions(); o.intra_op_num_threads = self.threads
        sess = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        dims = list(inp.shape)          # ints, or names/None for free dimensions
        if len(dims) != 3 or any(isinstance(d, int) and d != want for d, want in zip(dims[1:], (16, 96))):
            raise ValueError(f"input {dims}: an openWakeWord head takes (1, 16, 96)")
        out = sess.run(None, {inp.name: np.zeros((1, 16, 96), np.float32)})[0]
        if np.asarray(out).size != 1:
            raise ValueError(f"output of {np.asarray(out).size} values: a head gives one score")
        return sess, inp.name

    def _path(self, name: str) -> str | None:
        name = os.path.basename(name)
        for d in (self.custom_dir, self.builtin_dir):
            p = os.path.join(d, name)
            if name.endswith(".onnx") and name not in FEATURE_MODELS and os.path.isfile(p):
                return p
        return None

    def models(self) -> list[dict]:
        out = []
        for d, custom in ((self.builtin_dir, False), (self.custom_dir, True)):
            for f in sorted(os.listdir(d)):
                if f.endswith(".onnx") and f not in FEATURE_MODELS:
                    out.append({"file": f, "name": os.path.splitext(f)[0], "custom": custom,
                                "active": f == self.head_file})
        return out

    def set_model(self, name: str, persist: bool = True) -> None:
        path = self._path(name)
        if path is None:
            raise ValueError(f"no wake model {name}")
        self.head = self._open_head(path)   # one assignment: a score already running keeps the old pair
        self.head_file = os.path.basename(path)
        if persist:
            self._save()
        _LOG.info("wake model now %s", self.head_file)

    def add_model(self, filename: str, blob: bytes) -> str:
        name = os.path.basename(filename or "").strip()
        if not name.endswith(".onnx") or name in FEATURE_MODELS or name.startswith("."):
            raise ValueError("send an openWakeWord model as a .onnx file")
        if len(blob) > MAX_HEAD_BYTES:
            raise ValueError("the file is larger than 20 MB: not a wake word head")
        if os.path.exists(os.path.join(self.builtin_dir, name)):
            raise ValueError(f"{name} is a built-in model: rename the file")
        tmp = os.path.join(self.custom_dir, "." + name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(blob)
        try:
            self._open_head(tmp)
        except Exception as e:
            os.remove(tmp)
            raise ValueError(f"{name} is not a usable openWakeWord model: {e}") from None
        os.replace(tmp, os.path.join(self.custom_dir, name))
        _LOG.info("wake model %s added", name)
        return name

    def remove_model(self, name: str) -> None:
        name = os.path.basename(name)
        path = os.path.join(self.custom_dir, name)
        if not os.path.isfile(path):
            raise ValueError(f"{name} is not an uploaded model")
        if name == self.head_file:
            self.set_model(self.default_head)
        os.remove(path)
        _LOG.info("wake model %s removed", name)

    def new_stream(self) -> OwwStream:
        return OwwStream(self.features, self.stream_start)

    def step(self, stream: OwwStream, audio: np.ndarray) -> list[float]:
        """Add audio to a session's stream; one score per new 80 ms step."""
        k = stream.push(audio)
        head, head_in = self.head
        out = []
        for j in range(k - 1, -1, -1):
            e = stream.emb[len(stream.emb) - 16 - j:len(stream.emb) - j]
            r = head.run(None, {head_in: e[None].astype(np.float32)})[0]
            out.append(float(np.asarray(r).ravel()[0]))
        return out

    def score(self, audio: np.ndarray) -> float:
        e = self.features.embeddings(audio)
        if len(e) < 16:
            return 0.0
        head, head_in = self.head
        out = head.run(None, {head_in: e[-16:][None].astype(np.float32)})[0]
        return float(np.asarray(out).ravel()[0])

    def keep_rejected(self, sess: Session, pcm: np.ndarray) -> None:
        if not self.save_rejected or len(pcm) < RATE // 2:
            return
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(sess.started)) + f"_{max(sess.scores or [0]):.2f}.wav"
        with wave.open(os.path.join(self.rejected_dir, name), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(pcm.astype(np.int16).tobytes())
        sess.clip = name
        files = sorted(os.listdir(self.rejected_dir))
        for old in files[:-200]:                        # keep the last 200
            os.remove(os.path.join(self.rejected_dir, old))

    def program(self) -> WakeProgram:
        return WakeProgram(
            name="voice_hub", description="Voice Hub second-stage wake word", installed=True,
            attribution=Attribution(name="openWakeWord features (Apache-2.0)", url="https://github.com/dscripka/openWakeWord"),
            version="1", models=[WakeModel(
                name=self.model_name, description="Hey Jarvis", phrase="Hey Jarvis", installed=True,
                attribution=Attribution(name="Voice Hub", url=""), languages=["en"], version="1")])

    def summary(self) -> dict:
        per: dict[str, dict] = {}
        for s in self.sessions:
            d = per.setdefault(s.satellite or "unknown", {"confirmed": 0, "rejected": 0, "last": []})
            d["confirmed" if s.detected_at_ms is not None else "rejected"] += 1
            d["last"].append(s.as_dict())
        for d in per.values():
            d["last"] = list(reversed(d["last"][-10:]))
        return {"model": self.model_name, "head": os.path.splitext(self.head_file)[0], "models": self.models(),
                "threshold": self.threshold, "satellites": per}


class WakeHandler(AsyncEventHandler):
    def __init__(self, engine: WakeEngine, resolve, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.engine, self.resolve = engine, resolve
        self.conv = AudioChunkConverter(rate=RATE, width=2, channels=1)
        self.stream: OwwStream | None = None
        self.kept: list[np.ndarray] = []
        self.sess: Session | None = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(Info(wake=[self.engine.program()]).event())
            return True
        if Detect.is_type(event.type) or AudioStart.is_type(event.type):
            if self.sess is None:
                self.sess = Session(time.time(), os.path.splitext(self.engine.head_file)[0])
                self.stream = self.engine.new_stream(); self.kept = []
            return True
        if AudioChunk.is_type(event.type):
            if self.sess is None:
                self.sess = Session(time.time(), os.path.splitext(self.engine.head_file)[0])
            if self.stream is None:
                self.stream = self.engine.new_stream()
            chunk = self.conv.convert(AudioChunk.from_event(event))
            x = np.frombuffer(chunk.audio, np.int16)
            self.kept.append(x)
            if sum(len(k) for k in self.kept) > KEEP_S * RATE:
                self.kept.pop(0)
            if self.sess.detected_at_ms is None:
                scores = await asyncio.get_running_loop().run_in_executor(None, self.engine.step, self.stream, x)
                for s in scores:
                    self.sess.scores.append(s)
                    if s >= self.engine.threshold:
                        ms = int(1000 * sum(len(k) for k in self.kept) / RATE)
                        self.sess.detected_at_ms = ms
                        await self.write_event(Detection(name=self.engine.model_name, timestamp=ms).event())
                        _LOG.info("wake CONFIRMED (%.2f >= %.2f) at %d ms", s, self.engine.threshold, ms)
                        break
            return True
        if AudioStop.is_type(event.type):
            await self._finish()
            return True
        return True

    async def disconnect(self) -> None:
        await self._finish()

    async def _finish(self) -> None:
        sess, self.sess = self.sess, None
        if sess is None:
            return
        if sess.detected_at_ms is None:
            try:
                await self.write_event(NotDetected().event())
            except Exception:
                pass
            pcm = np.concatenate(self.kept) if self.kept else np.zeros(0, np.int16)
            self.engine.keep_rejected(sess, pcm)
            _LOG.info("wake rejected (max %.2f < %.2f)", max(sess.scores or [0]), self.engine.threshold)
        self.engine.sessions.append(sess)
        asyncio.create_task(self.resolve(sess))

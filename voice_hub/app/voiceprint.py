"""Voiceprints (speaker embeddings) and the registry of enrolled people.

Model: NVIDIA NeMo TitaNet large (CC-BY-4.0) through sherpa-onnx, 113 MB, CPU
only. Chosen on 2026-10-19 over WeSpeaker ResNet34 (VoxCeleb): on household
commands through an Echo Dot the old model put every voice at 0.8-0.9 (equal
error rate 37% on the enrolled samples, median gap between the two people
0.03), TitaNet large 1.9% and 0.22.
The registry records which model its voiceprints come from; with another
model they are recomputed from the saved recordings (Hub.reembed).

Privacy: enrollment audio is only used to recognise who is speaking. Never use
it to train a wake word model: the data folder is marked to make that explicit.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field

import numpy as np
import sherpa_onnx

RATE = 16000


@dataclass
class Settings:
    # Shorter commands give an unreliable voiceprint: answer "unknown".
    min_duration_s: float = 0.8
    # Minimum similarity to name someone, and minimum gap between the best and
    # second-best person, on TitaNet's plain cosine similarity (2026-10-19, 21
    # enrolled samples: same person 0.50-0.65, other person 0.28-0.38).
    threshold: float = 0.35
    margin: float = 0.10
    # Subtract the mean of recent commands (Cohort) before comparing. OFF: with
    # one person giving most commands that mean IS their voice, and removing it
    # left 9% of commands named instead of 44% (TitaNet, 2026-10-19).
    use_cohort: bool = False
    # With the cohort, until it is big enough: plain cosine similarity.
    raw_threshold: float = 0.35
    raw_margin: float = 0.10


class Cohort:
    """Voiceprints of recent commands from any speaker (numbers only, no audio).

    Every satellite and room leaves the same mark on every voice, and the
    speaker model hears it: two people through the same Echo scored 0.85 on
    their raw means. The mean of many commands is that mark; subtracting it
    before comparing leaves what differs between people (measured on our
    commands: the two means went from +0.85 to -0.87, leave-one-out 18 -> 19
    of 21, margins 0.1 -> 0.4-1.1)."""

    MIN = 20
    MAX = 1000

    def __init__(self, folder: str) -> None:
        self.file = os.path.join(folder, "cohort.npy")
        self._lock = threading.Lock()
        try:
            self.items = list(np.load(self.file))
        except (OSError, ValueError):
            self.items = []

    def add(self, embedding: np.ndarray) -> None:
        with self._lock:
            self.items.append(np.asarray(embedding, dtype=np.float32))
            del self.items[:-self.MAX]
            np.save(self.file, np.array(self.items, dtype=np.float32))

    def center(self) -> np.ndarray | None:
        with self._lock:
            if len(self.items) < self.MIN:
                return None
            return np.mean(np.array(self.items), axis=0)


@dataclass
class Result:
    person: str            # a name, or "unknown"
    score: float           # cosine similarity with the chosen person
    runner_up: float       # similarity of the second-best person
    duration_s: float
    reason: str = ""       # why "unknown", when it is
    scores: dict = field(default_factory=dict)


class Voiceprint:
    def __init__(self, model: str, threads: int = 2) -> None:
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model, num_threads=threads)
        if not cfg.validate():
            raise ValueError(f"invalid model: {model}")
        self._ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self._lock = threading.Lock()

    def compute(self, pcm16: bytes) -> np.ndarray:
        """Mono 16 kHz int16 PCM -> unit-length embedding."""
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            s = self._ext.create_stream()
            s.accept_waveform(RATE, x)
            s.input_finished()
            v = np.array(self._ext.compute(s), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


class Registry:
    """people.json: {"version": 2, "people": {name: {"embeddings": [...], "sources": [...]}}}.
    `sources` are the recordings each voiceprint came from (for the web page).
    Compared against each person's mean. Version 1 files ({name: [embeddings]}) still load."""

    MARKER = "VOICE-RECOGNITION-ONLY"
    TOP_K = 3

    # Each person is compared with their samples from the same satellite when
    # they have at least this many, otherwise with all: every satellite leaves
    # its own mark on a voice.
    MIN_PER_DEVICE = 3

    def __init__(self, folder: str, settings: Settings | None = None, model: str = "") -> None:
        self.folder = folder
        self.model = model                 # model the voiceprints belong to
        self.file_model = ""               # model named in people.json (old files: none)
        self._device_of: dict[str, str | None] = {}
        self.settings = settings or Settings()
        self.cohort = Cohort(folder)
        os.makedirs(folder, exist_ok=True)
        marker = os.path.join(folder, self.MARKER)
        if not os.path.exists(marker):
            with open(marker, "w") as f:
                f.write("Household voices: only for recognising who is speaking. "
                        "Never use them to train a wake word.\n")
        self.file = os.path.join(folder, "people.json")
        self.people: dict[str, list[list[float]]] = {}
        self.sources: dict[str, list[str]] = {}
        self._mtime = None
        self.reload_if_changed()

    def reload_if_changed(self) -> None:
        """Pick up people enrolled by another process (enroll.py) without a restart."""
        try:
            mtime = os.path.getmtime(self.file)
        except FileNotFoundError:
            return
        if mtime != self._mtime:
            with open(self.file) as f:
                d = json.load(f)
            self.file_model = d.get("model", "wespeaker_en_voxceleb_resnet34_LM") if d.get("version") == 2 else ""
            if d.get("version") == 2:
                self.people = {n: p["embeddings"] for n, p in d["people"].items()}
                self.sources = {n: p.get("sources", []) for n, p in d["people"].items()}
            else:
                self.people, self.sources = d, {n: [""] * len(v) for n, v in d.items()}
            self._mtime = mtime
            if self._dedupe():
                self.save()

    def _dedupe(self) -> bool:
        """The same recording enrolled twice for one person (e.g. Save pressed
        three times) is kept once."""
        changed = False
        for n in list(self.people):
            seen, emb, src = set(), [], []
            for e, f in zip(self.people[n], self.sources.get(n, [""] * len(self.people[n]))):
                if f and f in seen:
                    changed = True
                    continue
                seen.add(f)
                emb.append(e)
                src.append(f)
            self.people[n], self.sources[n] = emb, src
        return changed

    def save(self) -> None:
        tmp = self.file + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"version": 2, "model": self.model, "people": {
                n: {"embeddings": e, "sources": self.sources.get(n, [""] * len(e))}
                for n, e in self.people.items()}}, f)
        os.replace(tmp, self.file)
        self._mtime = os.path.getmtime(self.file)

    def add(self, name: str, embedding: np.ndarray, source: str = "") -> int:
        self.reload_if_changed()
        if source and source in self.sources.get(name, []):
            return len(self.people[name])
        self.people.setdefault(name, []).append([float(v) for v in embedding])
        self.sources.setdefault(name, [""] * (len(self.people[name]) - 1)).append(source)
        self.save()
        return len(self.people[name])

    def remove(self, name: str) -> bool:
        self.reload_if_changed()
        if self.people.pop(name, None) is None:
            return False
        self.sources.pop(name, None)
        self.save()
        return True

    def remove_source(self, name: str, source: str) -> bool:
        """Drop the voiceprint that came from one recording."""
        self.reload_if_changed()
        src = self.sources.get(name, [])
        if source not in src:
            return False
        i = src.index(source)
        del src[i]
        del self.people[name][i]
        if not self.people[name]:
            del self.people[name]
            self.sources.pop(name, None)
        self.save()
        return True

    @staticmethod
    def _unit(v: np.ndarray, center: np.ndarray | None) -> np.ndarray:
        if center is not None:
            v = v - center
        return v / (np.linalg.norm(v) + 1e-9)

    def means(self, center: np.ndarray | None = None) -> dict[str, np.ndarray]:
        out = {}
        for name, items in self.people.items():
            E = np.array(items, dtype=np.float32)
            m = np.mean([self._unit(e, center) for e in E], axis=0)
            out[name] = m / (np.linalg.norm(m) + 1e-9)
        return out

    @property
    def stale(self) -> bool:
        """The voiceprints were computed by another model."""
        return bool(self.people) and bool(self.model) and self.file_model != self.model

    def device_of(self, source: str) -> str | None:
        """Satellite a saved recording came from (its collection JSON)."""
        if source not in self._device_of:
            try:
                with open(os.path.join(self.folder, "collection", source + ".json")) as f:
                    self._device_of[source] = json.load(f).get("device_id")
            except (OSError, ValueError):
                self._device_of[source] = None
        return self._device_of[source]

    def identify(self, embedding: np.ndarray, duration_s: float, device_id: str | None = None) -> Result:
        s = self.settings
        self.reload_if_changed()
        center = self.cohort.center() if s.use_cohort else None
        if s.use_cohort and center is None:
            threshold, margin = s.raw_threshold, s.raw_margin
        else:
            threshold, margin = s.threshold, s.margin
        means = self.means(center)
        if not means:
            return Result("unknown", 0.0, 0.0, duration_s, "nobody enrolled")
        x = self._unit(np.asarray(embedding, dtype=np.float32), center)
        # Each person scores the mean of their TOP_K most similar samples, not
        # their overall mean: a command said over music meets that person's
        # samples said over music. Measured on our commands: same 19/21 as the
        # mean, but the closest correct call went from a 0.02 to a 0.13 margin.
        scores = {}
        for n, items in self.people.items():
            if device_id:
                same = [e for e, f in zip(items, self.sources.get(n, [])) if f and self.device_of(f) == device_id]
                if len(same) >= self.MIN_PER_DEVICE:
                    items = same
            sims = np.sort([float(np.dot(x, self._unit(np.asarray(e, dtype=np.float32), center))) for e in items])
            scores[n] = float(np.mean(sims[-self.TOP_K:])) if len(sims) >= self.TOP_K else float(np.dot(x, means[n]))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        name, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        r = Result(name, best, second, duration_s, scores=scores)
        if duration_s < s.min_duration_s:
            r.person, r.reason = "unknown", f"command too short ({duration_s:.1f} s)"
        elif best < threshold:
            r.person, r.reason = "unknown", f"low similarity ({best:.2f} < {threshold})"
        elif best - second < margin:
            r.person, r.reason = "unknown", f"too close to call ({best:.2f} vs {second:.2f})"
        return r

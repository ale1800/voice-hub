"""openWakeWord feature pipeline (melspectrogram + Google speech embedding),
re-implemented on onnxruntime so the add-on does not need the openwakeword
package. Matches openwakeword.utils.AudioFeatures (checked against 0.4.0):
mel = melspec(int16 audio as float32) / 10 + 2; one 96-d embedding per
76 mel frames, hop 8 frames (80 ms)."""
import numpy as np
import onnxruntime as ort


class OwwFeatures:
    def __init__(self, melspec_path: str, embedding_path: str, threads: int = 1):
        o = ort.SessionOptions(); o.intra_op_num_threads = threads; o.inter_op_num_threads = 1
        self.mel = ort.InferenceSession(melspec_path, o, providers=["CPUExecutionProvider"])
        self.emb = ort.InferenceSession(embedding_path, o, providers=["CPUExecutionProvider"])
        self.mel_in = self.mel.get_inputs()[0].name
        self.emb_in = self.emb.get_inputs()[0].name

    def melspec(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, np.float32)[None, :]
        return np.squeeze(self.mel.run(None, {self.mel_in: x})[0]) / 10.0 + 2.0

    def embeddings(self, audio: np.ndarray) -> np.ndarray:
        """int16 mono 16 kHz -> (T, 96)."""
        m = self.melspec(audio)
        if len(m) < 76:
            return np.zeros((0, 96), np.float32)
        win = np.stack([m[i:i + 76] for i in range(0, len(m) - 75, 8)])[..., None].astype(np.float32)
        return np.squeeze(self.emb.run(None, {self.emb_in: win})[0], axis=(1, 2)).astype(np.float32)


class OwwStream:
    """The same features computed as audio arrives, as openWakeWord's
    AudioFeatures._streaming_features does: per 1280 new samples, the mel
    spectrogram of only those (plus 3 hops of context) and ONE new embedding.
    Recomputing the whole 2 s window instead cost 270 ms per 80 ms step on a
    Home Assistant box (2026-10-19): 3.4x slower than real time, so the second
    stage confirmed 3 s late and the command after the wake word was lost."""

    HOP = 1280

    def __init__(self, feats: OwwFeatures, start: np.ndarray | None = None) -> None:
        self.f = feats
        self.raw = np.zeros(0, np.int16)
        self.pending = np.zeros(0, np.int16)
        self.mel = np.ones((76, 32), np.float32)          # openWakeWord's starting buffer
        # openWakeWord starts from the embeddings of 4 s of noise, so a score can
        # be given at once; fixed seed here, so runs are repeatable.
        if start is None:
            start = self.start(feats)
        self.emb = start.copy()

    @staticmethod
    def start(feats: OwwFeatures) -> np.ndarray:
        """The starting embeddings (computed once: ~0.5 s on a Home Assistant box)."""
        noise = np.random.default_rng(0).integers(-1000, 1000, 16000 * 4).astype(np.int16)
        return feats.embeddings(noise)[-16:]

    def push(self, audio: np.ndarray) -> int:
        """Add int16 audio; returns how many new embeddings (80 ms steps) it gave."""
        self.pending = np.concatenate([self.pending, np.asarray(audio, np.int16)])
        n = (len(self.pending) // self.HOP) * self.HOP
        if n == 0:
            return 0
        chunk, self.pending = self.pending[:n], self.pending[n:]
        self.raw = np.concatenate([self.raw, chunk])[-16000 * 10:]
        self.mel = np.vstack([self.mel, self.f.melspec(self.raw[-n - 160 * 3:])])[-970:]
        new = []
        for i in range(n // self.HOP - 1, -1, -1):
            end = len(self.mel) - 8 * i
            w = self.mel[end - 76:end]
            if len(w) == 76:
                new.append(w)
        if new:
            e = self.f.emb.run(None, {self.f.emb_in: np.stack(new)[..., None].astype(np.float32)})[0]
            self.emb = np.vstack([self.emb, np.squeeze(e, axis=(1, 2)).astype(np.float32)])[-64:]
        return len(new)

    def last16(self) -> np.ndarray:
        return self.emb[-16:]

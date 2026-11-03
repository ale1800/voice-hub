"""Enroll, list, remove or test people in the voice-hub registry.

  python enroll.py add Alice /data/collection/20261012-101500.wav ...
  python enroll.py list
  python enroll.py remove Alice
  python enroll.py test file.wav           # who would it be, without enrolling

Enroll from REAL commands that went through your satellite (saved in
<data>/collection/ when "collect" is on), not from a phone microphone: a
different microphone and audio chain shifts the voiceprint more than the
difference between two people.
"""
import argparse
import os
import wave

from voiceprint import RATE, Registry, Voiceprint

DEFAULT_MODEL = os.environ.get("VOICE_HUB_MODEL", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "nemo_en_titanet_large.onnx"))


def read(fn: str) -> tuple[bytes, float]:
    with wave.open(fn) as w:
        if (w.getframerate(), w.getsampwidth(), w.getnchannels()) != (RATE, 2, 1):
            raise SystemExit(f"{fn}: needs a mono 16 kHz 16-bit WAV")
        pcm = w.readframes(w.getnframes())
    return pcm, len(pcm) / 2 / RATE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["add", "list", "remove", "test"])
    p.add_argument("args", nargs="*")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default=os.environ.get("VOICE_HUB_DATA", "data"))
    a = p.parse_args()
    reg = Registry(a.data, model=os.path.splitext(os.path.basename(a.model))[0])
    if a.action == "list":
        for n, items in reg.people.items():
            print(f"{n}: {len(items)} voiceprints")
        return
    if a.action == "remove":
        print("removed" if reg.remove(a.args[0]) else "not enrolled")
        return
    vp = Voiceprint(a.model)
    if a.action == "test":
        for fn in a.args:
            pcm, d = read(fn)
            r = reg.identify(vp.compute(pcm), d)
            print(f"{os.path.basename(fn)}: {r.person} ({r.score:.2f}, runner-up {r.runner_up:.2f}) {r.reason}")
        return
    name, files = a.args[0], a.args[1:]
    for fn in files:
        pcm, d = read(fn)
        n = reg.add(name, vp.compute(pcm), os.path.basename(fn))
        print(f"{name}: +{os.path.basename(fn)} ({d:.1f} s), now {n} voiceprints")


if __name__ == "__main__":
    main()

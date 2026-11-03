"""Plays the part of Assist: sends a WAV to the service like the pipeline would.

  .venv/bin/python prova_client.py file.wav [tcp://127.0.0.1:10310]
"""
import asyncio
import sys
import time
import wave

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient


async def main(fn: str, uri: str) -> None:
    with wave.open(fn) as w:
        rate, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        pcm = w.readframes(w.getnframes())
    t0 = time.monotonic()
    async with AsyncClient.from_uri(uri) as c:
        await c.write_event(Transcribe(language="it").event())
        await c.write_event(AudioStart(rate=rate, width=width, channels=ch).event())
        passo = rate * width * ch // 10  # blocchi da 100 ms, come un satellite
        for i in range(0, len(pcm), passo):
            await c.write_event(AudioChunk(rate=rate, width=width, channels=ch, audio=pcm[i:i + passo]).event())
        await c.write_event(AudioStop().event())
        while True:
            e = await c.read_event()
            if e is None:
                break
            if Transcript.is_type(e.type):
                print(f"\"{Transcript.from_event(e).text}\" in {(time.monotonic() - t0) * 1000:.0f} ms")
                break


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "tcp://127.0.0.1:10310"))

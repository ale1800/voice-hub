"""voice-hub entry point.

Two ways to get the command audio, both feeding hub.Hub:

  * capture (default): listens to the Assist pipeline inside Home Assistant.
    Nothing to change in your pipelines; see capture.py. The result usually
    arrives while the cloud is still transcribing, but it is a race: the log
    says for every command whether it beat the intent stage.
  * Wyoming STT pass-through (--wyoming): add the Wyoming integration, pick
    "voice-hub" as speech-to-text in a pipeline. It forwards the audio to the
    real STT entity through Home Assistant's /api/stt and only returns the text
    AFTER publishing the speaker, so sentence automations always see it.
    Useful for satellites whose pipeline you can change but whose firmware you
    can't, or when you need the guarantee.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from functools import partial

import aiohttp
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from wyoming.handle import Handled

from capture import Capture
from hub import Hub, wav_from_pcm
from voiceprint import RATE, Settings
from web import Web

_LOG = logging.getLogger("voice-hub")
VERSION = "0.10.1"
ATTRIBUTION = Attribution(name="LocalDot voice-hub", url="https://github.com/ale1800/voice-hub")


async def transcribe(hub: Hub, pcm: bytes, language: str) -> str:
    fmt = f"format=wav; codec=pcm; sample_rate={RATE}; bit_rate=16; channel=1; language={language}"
    async with hub.session.post(f"{hub.ha_url}/api/stt/{hub.args.stt_entity}", data=wav_from_pcm(pcm),
                                headers={**hub.headers(), "X-Speech-Content": fmt},
                                timeout=aiohttp.ClientTimeout(total=30)) as r:
        d = await r.json(content_type=None)
        if d.get("result") != "success":
            _LOG.warning("STT %s returned %s", hub.args.stt_entity, d)
        return d.get("text") or ""


class WyomingHandler(AsyncEventHandler):
    def __init__(self, hub: Hub, *a, **k) -> None:
        super().__init__(*a, **k)
        self.hub = hub
        self.converter = AudioChunkConverter(rate=RATE, width=2, channels=1)
        self.language = hub.args.language   # empty = Home Assistant's, resolved when needed
        self.audio = bytearray()

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            lang = (await self.hub.language()).split("-")[0]
            await self.write_event(Info(asr=[AsrProgram(
                name="voice-hub", description="Speech-to-text pass-through that recognises the speaker",
                attribution=ATTRIBUTION, installed=True, version=VERSION,
                models=[AsrModel(name=f"via {self.hub.args.stt_entity}",
                                 description="forwards to Home Assistant STT",
                                 attribution=ATTRIBUTION, installed=True,
                                 languages=[lang], version=VERSION)])]).event())
            return True
        if Transcribe.is_type(event.type):
            t = Transcribe.from_event(event)
            if t.language and "-" in t.language:
                self.language = t.language
            return True
        if AudioStart.is_type(event.type):
            self.audio = bytearray()
            return True
        if AudioChunk.is_type(event.type):
            self.audio += self.converter.convert(AudioChunk.from_event(event)).audio
            return True
        if AudioStop.is_type(event.type):
            await self._finish()
            return False
        return True

    async def _finish(self) -> None:
        pcm = bytes(self.audio)
        t0 = time.monotonic()
        # STT and voiceprint run IN PARALLEL: identification must not delay the reply.
        stt = asyncio.create_task(transcribe(self.hub, pcm, self.language or await self.hub.language()))
        result, t_vp = await self.hub.identify(pcm)
        text = await stt
        await self.hub.publish(result, text=text)
        _LOG.info("wyoming: %.1f s | %s (%.2f, runner-up %.2f) %s | voiceprint %.0f ms, total %.0f ms | \"%s\"",
                  result.duration_s, result.person, result.score, result.runner_up, result.reason,
                  t_vp * 1000, (time.monotonic() - t0) * 1000, text)
        base = self.hub.collect(pcm, result, text=text, source="wyoming")
        self.hub.add_recent(result, base, text=text, source="wyoming", device_name="Wyoming")
        await self.write_event(Transcript(text=text).event())


class HubHandler(AsyncEventHandler):
    """The Wyoming service on 10400: second-stage wake word and/or conversation agent."""

    def __init__(self, engine, resolve, agent, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.agent = agent
        self.wake = None
        if engine is not None:
            from wake import WakeHandler
            self.wake = WakeHandler(engine, resolve, *args, **kwargs)
            self.wake.write_event = self.write_event      # one connection, one writer
        self.engine = engine

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(Info(wake=[self.engine.program()] if self.engine else [],
                                        handle=[self.agent.info()] if self.agent else []).event())
            return True
        if Transcript.is_type(event.type) and self.agent is not None:
            await self.write_event((await self.agent.handle(Transcript.from_event(event))).event())
            return True
        if self.wake is not None:
            return await self.wake.handle_event(event)
        return True

    async def disconnect(self) -> None:
        if self.wake is not None:
            await self.wake.disconnect()


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-capture", action="store_true", help="don't listen to the Assist pipeline")
    p.add_argument("--wyoming", action="store_true", help="also run the Wyoming STT pass-through")
    p.add_argument("--uri", default="tcp://0.0.0.0:10310")
    p.add_argument("--stt-entity", default="stt.home_assistant_cloud", help="Wyoming mode: STT entity to forward to")
    p.add_argument("--language", default="", help="empty = Home Assistant's language")
    p.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "..", "models",
                                                   "wespeaker_en_voxceleb_resnet34_LM.onnx"))
    p.add_argument("--data", default="data")
    p.add_argument("--collect", action="store_true", help="save every command to <data>/collection/")
    p.add_argument("--threshold", type=float, default=Settings.threshold)
    p.add_argument("--margin", type=float, default=Settings.margin)
    p.add_argument("--min-duration", type=float, default=Settings.min_duration_s)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--web-port", type=int, default=8099, help="web page (Home Assistant ingress); 0 = off")
    p.add_argument("--wake", action="store_true", help="also run the second-stage wake word engine (Wyoming)")
    p.add_argument("--wake-uri", default="tcp://0.0.0.0:10400")
    p.add_argument("--wake-models", default="/models/oww", help="melspectrogram.onnx, embedding_model.onnx and heads")
    p.add_argument("--wake-model", default="hey_jarvis_v0.1.onnx", help="the head: file in --wake-models or a path")
    p.add_argument("--wake-threshold", type=float, default=0.1)
    p.add_argument("--save-rejected", action="store_true", help="keep the audio of rejected wakes (last 200)")
    p.add_argument("--agent", action="store_true",
                   help="also offer Voice Hub as a conversation agent (on --wake-uri)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    hub = Hub(args)
    modes = [m for m, on in (("pipeline capture", not args.no_capture),
                             (f"Wyoming on {args.uri} -> {args.stt_entity}", args.wyoming),
                             (f"wake word on {args.wake_uri} ({args.wake_model})", args.wake),
                             (f"conversation agent on {args.wake_uri}", args.agent)) if on]
    _LOG.info("voice-hub %s | %s | enrolled: %s", VERSION, " + ".join(modes) or "nothing to do",
              ", ".join(hub.registry.people) or "nobody")
    async with aiohttp.ClientSession() as session:
        hub.session = session
        asyncio.get_running_loop().run_in_executor(None, hub.seed_cohort)
        tasks = []
        capture = None if args.no_capture else Capture(hub)
        if capture is not None:
            tasks.append(asyncio.create_task(capture.run_forever()))
        if args.wake:
            from wake import WakeEngine
            hub.wake = WakeEngine(args.wake_models, args.wake_model, args.wake_threshold, args.data,
                                  args.save_rejected, threads=1)
        if args.agent and capture is not None:
            from agent import Agent
            hub.agent = Agent(hub, capture, args.data)

        async def resolve(sess):
            await asyncio.sleep(3)
            if capture is not None:
                sess.satellite = await capture.satellite_at(sess.started)
        if hub.wake is not None or hub.agent is not None:
            wake_server = AsyncServer.from_uri(args.wake_uri)
            tasks.append(asyncio.create_task(wake_server.run(partial(HubHandler, hub.wake, resolve, hub.agent))))
        if args.web_port:
            await Web(hub, capture).run(args.web_port)
        if args.wyoming:
            server = AsyncServer.from_uri(args.uri)
            tasks.append(asyncio.create_task(server.run(partial(WyomingHandler, hub))))
        tasks.append(asyncio.create_task(asyncio.Event().wait()))   # keep the web page up
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())

# Voice Hub

Voice Hub tells Home Assistant **who is speaking** to your voice satellites.
You don't change your pipelines: your speech-to-text (Home Assistant Cloud,
Whisper, ...) stays exactly as it is.

## How it works

Every satellite already sends its command to Home Assistant, which passes it to
the speech-to-text engine. Voice Hub asks Home Assistant for a copy of that
audio through the official `assist_pipeline/device/capture` API (the one used
by the voice setup wizard):

1. when an `assist_satellite.*` entity turns `listening`, it starts listening
   to that device;
2. when the audio stops, it compares the voice with the people you enrolled,
   while the cloud is still transcribing;
3. it publishes the result:
   - event `voice_hub_speaker` with `person`, `score`, `runner_up`,
     `duration_s`, `reason`, `device_id`, `device_name`
   - entity `sensor.voice_hub_last_speaker` (state = name or `unknown`)

Identification uses the **command**, not the wake word: it is longer, and music
is already ducked while you speak it.

**It is a race with speech-to-text.** The result normally arrives before the
sentence is handled, but not always. The log says, for every command,
`published N ms BEFORE intent` or `AFTER intent`. In automations that must be
sure, wait for the event:

```yaml
- wait_for_trigger:
    - trigger: event
      event_type: voice_hub_speaker
  timeout: "00:00:01"
```

Notes:
- Home Assistant allows one capture per device: if you open the voice debug
  tool on the same satellite, it takes over for that command.
- Each command adds one "started recording audio" line to the logbook.

## The Voice Hub page

Open **Voice Hub** in the Home Assistant sidebar:

- **Satellites**: every Assist satellite, its state and pipeline, the last
  command it heard (text, who, how early the result was) and its statistics,
  grouped: echo cancellation, network, system (free memory...), intercom.
  Read-only: only the sensors the satellite exposes to Home Assistant.
- **People**: **＋ Add person** starts a guided enrollment (below). You can also
  pick from saved commands, add samples later, or remove a person or a single
  sample at any time.
- **Commands**: the latest commands with transcript, satellite, guess and
  timing. Play or delete each recording.

The page is only served through Home Assistant (ingress); the add-on refuses
direct connections.

## Optional: Wyoming pass-through

Turn on `wyoming` to also expose a Wyoming speech-to-text on port 10310 that
forwards to `stt_entity` and only returns the text **after** publishing the
speaker. Add it with the Wyoming integration (host `local-voice-hub`, port
`10310`) and pick it as speech-to-text in a pipeline. Use it when you need the
guarantee and don't mind changing the pipeline.

## Enrolling people

**Guided (recommended).** People → ＋ Add person, type the name, pick the
satellite, Start. The page shows 12 short questions one at a time with a
counter (1/12, 2/12...) and how to say each one (close, from 3–4 m, with music
playing, softly). Say the wake word and the question: when the transcript
matches the phrase it turns green and the next one appears; otherwise the page
shows what was heard and asks again (too short commands too). The last step is
free: your most frequent command. Redo or skip any step, listen, then Save.
About 5 minutes; it works even with `collect` off.

Why this script:
- **~30 s of speech in 10–15 commands.** Speaker recognition from short
  commands improves a lot between 3 s and 30 s of enrollment speech, and with
  more separate utterances ([Deep Speaker](https://arxiv.org/pdf/1705.03670),
  [GE2E study](https://arxiv.org/pdf/2011.04896)).
- **Commands, not read text.** With short test phrases, enrollment content
  similar to what is said later helps the most
  ([SRI, content matching](https://www.sri.com/publication/speech-natural-language-pubs/content-matching-for-short-duration-speaker-recognition/));
  Google moved Voice Match from repeating the wake word to full commands for
  the same reason. The questions still cover most sounds of the language.
- **Varied conditions, same satellite.** Distance, music and a tired voice change
  the voiceprint more than the words do, so the script asks for all of them,
  spoken to a real satellite (not a phone).
- All questions are harmless: they really go through Assist and get answered.
- Enroll everyone on each satellite type you have; redo children every few
  months.

**From saved commands.** Turn on `collect`: every command is saved to
`/share/voice_hub/collection/` (WAV + JSON with the device, the transcript and
the guess). Pick a person's commands from the page or the command line:

    docker exec addon_local_voice_hub python3 /app/enroll.py add Alice /share/voice_hub/collection/<file>.wav ...
    docker exec addon_local_voice_hub python3 /app/enroll.py list

## Options

| option | meaning |
|---|---|
| `capture` | listen to the Assist pipeline (default on) |
| `wyoming` | also run the Wyoming pass-through (default off) |
| `stt_entity` | Wyoming mode only: where to forward the audio |
| `language` | Empty (default) = Home Assistant's language. Sets the guided-enrollment phrases (English or Italian) and, in Wyoming mode, the STT language. Fill it only to use a different language than Home Assistant |
| `collect` | save every command for enrollment |
| `threshold` | minimum similarity to name someone (default 0.35) |
| `margin` | minimum gap between the best and second-best person (0.10) |
| `min_duration` | shorter commands are `unknown` (0.8 s) |
| `threads` | CPU cores for the voiceprint (2) |

**Speaker model.** NVIDIA NeMo TitaNet large (CC-BY-4.0, attribution in
`NOTICE.md`). It replaced WeSpeaker ResNet34 in 0.8.0: through an Echo Dot the
old model put every voice at 0.8-0.9 similarity and could hardly tell two
people apart. Enrolled samples are recomputed from their saved recordings on
the first start with a new model.

**Satellite by satellite.** Each satellite leaves its own mark on a voice.
When a person has at least 3 samples from the satellite that heard the
command, Voice Hub compares with those only, so enroll each person once from
each satellite.

## Second-stage wake word

With `wake` on, Voice Hub is also a Wyoming wake word engine (port 10400) that a
LocalDot can ask to double-check its own wake. The page shows, per satellite,
how many wakes were confirmed and rejected, and the rejected audio.

**Your own model.** Under *Second-stage wake word*, **Add a model…** takes any
openWakeWord model as `.onnx` (openWakeWord publishes each model in `.tflite`
and `.onnx`; only `.onnx` works here). It is checked before it is kept, then
picked from the *Model* menu; the change applies to the next check. Home
Assistant keeps seeing the same wake word name, so no pipeline has to change.
Uploaded models live in `/share/voice_hub/wake_models/`.

## Voice Hub as your conversation agent

On its own Voice Hub races the command: the name is published while Home
Assistant transcribes, and on a fast satellite the command can run first.
To be sure the name is there **before** the command runs, pick **Voice Hub**
as the pipeline's conversation agent (Settings → Voice assistants). It comes
from Home Assistant's own Wyoming integration, nothing to install.

Voice Hub then finishes the voiceprint as soon as the transcript arrives (at
most 1.2 s), publishes who spoke, and passes the command to the agent chosen
on the **Agent** tab - Home Assistant, OpenAI, Google, any agent you have - with
the same device, so "turn on the light" still means this room. The agent can
differ per satellite or per person, and other agents can be told who is
speaking (the Home Assistant agent never is: its sentences would stop matching).

When an LLM agent ends its reply with a question, Voice Hub reopens listening
on that satellite once the reply has played, and the answer continues the
same conversation.

Option `agent` (on by default) offers the agent; it does nothing until a
pipeline uses it.

## Calling a LocalDot

A LocalDot with the VoIP Stack intercom shows a **Call** button (in Home
Assistant's language) while it is online. It opens VoIP Stack's own browser
phone with that LocalDot as the destination and places the call; the phone
stays open until you hang up. *In call* shows while a call is up.

## Limits

- Works best with a few people and clear speech. Very similar voices, loud
  music and children's voices are harder.
- **Not a security feature**: a recording of someone's voice fools it.
- Enrollment audio is for speaker recognition only. Do not use it to train
  wake word models.

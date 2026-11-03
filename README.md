# Voice Hub

A Home Assistant add-on that tells Home Assistant **who is speaking** to your
Assist voice satellites. It listens to the command audio inside Home Assistant,
with no changes to your pipelines, and recognises the voice of the people you
enroll. Each satellite gets a sensor with the last speaker, which you can use in
automations and conversation agents.

It also offers:

- guided enrollment from its sidebar page (Italian or English phrases, following
  Home Assistant's language);
- an optional second-stage wake word check (openWakeWord) for satellites that ask for it;
- a conversation agent that waits for the speaker's name and passes the command on
  to any other agent;
- extras for LocalDot satellites: LED ring designs, a custom on-device wake word
  and a Call button for the VoIP intercom.

## Install

1. In Home Assistant open **Settings → Add-ons → Add-on Store**.
2. Open the **⋮** menu → **Repositories**, add
   `https://github.com/ale1800/voice-hub` and close the dialog.
3. Find **Voice Hub** in the store, click **Install**, then **Start**.
4. Open **Voice Hub** from the sidebar and use **Add person** to enroll someone.

Leave the `language` option empty to follow Home Assistant's language.
Supported architectures: `aarch64`, `amd64`.

Full documentation: [voice_hub/DOCS.md](voice_hub/DOCS.md).
Changes: [voice_hub/CHANGELOG.md](voice_hub/CHANGELOG.md).

## Privacy

Voice recognition runs on your Home Assistant. Voiceprints and enrolled
recordings stay in `/share/voice_hub` on your machine, and Voice Hub talks only
to Home Assistant itself. Speech-to-text goes through whatever engine your
Assist pipeline already uses. The speaker model is downloaded once, when the
add-on is built.

## License

MIT, see [LICENSE](LICENSE). The bundled and downloaded models keep their own
licenses ([voice_hub/NOTICE.md](voice_hub/NOTICE.md)); in particular the bundled
openWakeWord `hey_jarvis` model is CC BY-NC-SA 4.0 (non-commercial).

"""Guided enrollment: the page shows one phrase at a time, the person says the
wake word and the phrase to a satellite, and each command is taken as the
sample for the current step.

Why these phrases (see DOCS.md): short-command speaker recognition works best
with ~30 s of enrollment speech spread over 10-15 utterances, with content
close to what will really be said (commands and questions, not read text), and
with the same variety of conditions the satellite will hear later (distance,
music, tired voice). All phrases are harmless questions: they really go
through Assist and get answered.
"""
from __future__ import annotations

import difflib
import re
import time
import unicodedata
from dataclasses import dataclass, field

# The last step has no fixed text: shown as an instruction in the page language.
OWN = ("Your most frequent command", "The one you say most often, the way you usually say it, softly.")

CLOSE, FAR, MUSIC, QUIET = (
    "Stand close to the satellite, normal voice.",
    "Stand 3–4 m away, normal voice.",
    "Play some music on the satellite first, then speak normally.",
    "Close by, speak softly, like late in the evening.",
)

SCRIPTS = {
    "it": [
        ("Che ore sono adesso?", CLOSE),
        ("Che tempo farà domani pomeriggio?", CLOSE),
        ("Quanti gradi ci sono in bagno?", CLOSE),
        ("A che ora tramonta il sole a luglio?", CLOSE),
        ("Che giorno della settimana è oggi?", FAR),
        ("Quando scende la temperatura sotto zero?", FAR),
        ("Che cosa c'è in calendario venerdì sera?", FAR),
        ("Com'è la qualità dell'aria fuori?", MUSIC),
        ("Qual è la temperatura in soggiorno?", MUSIC),
        ("Quanto manca a mezzanotte?", MUSIC),
        ("Che ore sono?", QUIET),
        OWN,
    ],
    "en": [
        ("What time is it now?", CLOSE),
        ("What's the weather like tomorrow afternoon?", CLOSE),
        ("How warm is it in the bathroom?", CLOSE),
        ("What time does the sun set in July?", CLOSE),
        ("What day of the week is it today?", FAR),
        ("When will the temperature drop below zero?", FAR),
        ("What's on the calendar on Friday evening?", FAR),
        ("How is the air quality outside?", MUSIC),
        ("What's the temperature in the living room?", MUSIC),
        ("How long until midnight?", MUSIC),
        ("What time is it?", QUIET),
        OWN,
    ],
}

IDLE_TIMEOUT_S = 15 * 60
MATCH_RATIO = 0.7      # how close the transcript must be to the phrase


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", t).split())


def matches(phrase: str, heard: str) -> bool:
    a, b = normalize(phrase), normalize(heard)
    return bool(b) and (a in b or difflib.SequenceMatcher(None, a, b).ratio() >= MATCH_RATIO)


def script_for(language: str) -> list[tuple[str, str]]:
    return SCRIPTS.get(language.split("-")[0].lower(), SCRIPTS["en"])


@dataclass
class Step:
    phrase: str
    hint: str
    file: str | None = None
    entry: dict | None = None        # row in hub.recent (gets the transcript later)
    rejected: str | None = None      # why the last attempt was not taken
    skipped: bool = False
    pending: tuple | None = None     # (file, entry) waiting for the transcript
    free: bool = False               # any words are fine (the person's own command)


@dataclass
class Session:
    name: str
    device_id: str | None            # None = any satellite
    device_name: str
    steps: list[Step]
    current: int = 0
    touched: float = field(default_factory=time.monotonic)

    def advance(self) -> None:
        """Go to the first step still to do (after a redo, earlier answers are kept)."""
        self.current = next((i for i, st in enumerate(self.steps) if not st.file and not st.skipped),
                            len(self.steps))

    def as_dict(self) -> dict:
        return {"name": self.name, "device_id": self.device_id, "device_name": self.device_name,
                "current": self.current, "total": len(self.steps), "done": self.current >= len(self.steps),
                "steps": [{"phrase": s.phrase, "hint": s.hint, "file": s.file, "rejected": s.rejected,
                           "skipped": s.skipped, "checking": s.pending is not None,
                           "text": (s.entry or {}).get("text"),
                           "duration_s": (s.entry or {}).get("duration_s")} for s in self.steps]}


class Enrollment:
    def __init__(self, language: str, min_duration: float) -> None:
        self.language = language
        self.min_duration = min_duration
        self.session: Session | None = None

    def active(self) -> Session | None:
        s = self.session
        if s is not None and time.monotonic() - s.touched > IDLE_TIMEOUT_S:
            self.session = None
        return self.session

    def start(self, name: str, device_id: str | None, device_name: str, language: str | None = None) -> Session:
        # 2026-11-02: the phrases follow Home Assistant's language unless the add-on option sets one
        self.session = Session(name, device_id, device_name,
                               [Step(p, h, free=(p, h) == OWN) for p, h in script_for(language or self.language or "en")])
        return self.session

    def wants(self, device_id: str) -> bool:
        """Should a command from this satellite be saved for enrollment?"""
        s = self.active()
        return s is not None and s.current < len(s.steps) and s.device_id in (None, device_id)

    def offer(self, device_id: str, file: str | None, entry: dict, duration_s: float) -> None:
        """A command finished on a satellite: hold it for the current step until
        the transcript says it was the right phrase (confirm)."""
        if not self.wants(device_id):
            return
        s = self.session
        step = s.steps[s.current]
        s.touched = time.monotonic()
        if file is None:
            step.rejected = "the recording could not be saved"
        elif duration_s < self.min_duration:
            step.rejected = f"too short ({duration_s:.1f} s): say the whole phrase after the wake word"
        else:
            step.pending, step.rejected = (file, entry), None

    def confirm(self, file: str | None, text: str | None) -> None:
        """The transcript of a held command arrived: accept it or ask again."""
        s = self.active()
        if s is None or file is None:
            return
        step = next((st for st in s.steps if st.pending and st.pending[0] == file), None)
        if step is None:
            return
        (file, entry), step.pending = step.pending, None
        s.touched = time.monotonic()
        # No transcript (lookup failed): accept on duration alone rather than block.
        if text is None or step.free or matches(step.phrase, text):
            step.file, step.entry, step.rejected, step.skipped = file, entry, None, False
            s.advance()
        else:
            step.rejected = f"heard «{text}»" if text else "nothing was understood"

    def goto(self, index: int) -> None:
        """Redo a step: the next command replaces its sample."""
        s = self.active()
        if s is not None and 0 <= index < len(s.steps):
            s.steps[index].skipped = False
            s.current = index
            s.touched = time.monotonic()

    def skip(self) -> None:
        s = self.active()
        if s is not None and s.current < len(s.steps):
            step = s.steps[s.current]
            step.rejected = None
            step.skipped = not step.file      # skipping a redo keeps the earlier sample
            s.advance()
            s.touched = time.monotonic()

    def cancel(self) -> None:
        self.session = None

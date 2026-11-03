"""Custom on-device wake word for LocalDot boards (microWakeWord .tflite + .json).

The board runs the model itself (firmware: `localdot_ww_custom.yaml`); Voice Hub only
checks the two files and sends the model in pieces through the board's ESPHome actions
(ww_start / ww_chunk / ww_finish), so nothing else on the network is involved. While a
custom model is active the board switches its own "Hey Jarvis" models off.
"""
import base64
import json
import zlib

MAX_SIZE = 300 * 1024
CHUNK = 3000            # bytes per action call (4 kB of base64)


def status_of(device_entities: list[dict], states: dict) -> str | None:
    """State of the board's «Custom wake word» sensor, or None if the firmware has no support."""
    for ent in device_entities:
        eid = ent["entity_id"]
        if eid.startswith("sensor.") and eid.endswith("_custom_wake_word"):
            return (states.get(eid) or {}).get("state")
    return None


def check(model: bytes, manifest: bytes) -> dict:
    """Validates the files; returns the parameters for ww_finish. Raises ValueError with a readable reason."""
    if len(model) < 1024 or len(model) > MAX_SIZE:
        raise ValueError(f"The .tflite file must be between 1 kB and {MAX_SIZE // 1024} kB.")
    if model[4:8] != b"TFL3":
        raise ValueError("The .tflite file is not a TensorFlow Lite model.")
    try:
        m = json.loads(manifest)
        micro = m["micro"]
        word = str(m["wake_word"]).strip()
        cutoff = float(micro["probability_cutoff"])
        window = int(micro["sliding_window_size"])
        arena = int(micro.get("tensor_arena_size", 30000))
        step = int(micro.get("feature_step_size", 10))
    except (ValueError, KeyError, TypeError):
        raise ValueError("The .json file is not a microWakeWord manifest (wake_word, micro.probability_cutoff, "
                         "micro.sliding_window_size).")
    if m.get("type", "micro") != "micro":
        raise ValueError("Only microWakeWord models (type «micro») run on the board.")
    if step != 10:
        raise ValueError(f"The model uses feature_step_size {step}; the board needs 10.")
    if not word or len(word) > 40:
        raise ValueError("The wake word name is empty or too long.")
    if not (0.05 <= cutoff <= 0.99) or not (1 <= window <= 30) or not (8000 <= arena <= 250000):
        raise ValueError("probability_cutoff, sliding_window_size or tensor_arena_size out of range.")
    return {"wake_word": word, "cutoff": cutoff, "window": window, "arena": arena}


def pieces(model: bytes):
    """(start args, [chunk args...]) for the board's actions."""
    start = {"size": len(model), "crc32": f"{zlib.crc32(model) & 0xffffffff:08x}"}
    chunks = [{"offset": o, "data": base64.b64encode(model[o:o + CHUNK]).decode()}
              for o in range(0, len(model), CHUNK)]
    return start, chunks

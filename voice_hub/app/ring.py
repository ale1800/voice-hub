"""LED ring designs for LocalDot boards (one per occasion), sent to the board.

The board stores each design in flash and uses it on its own, also when Voice Hub
is not running (firmware: `localdot_ring_events.yaml`). Voice Hub keeps a copy
here only to show and edit it. Occasions, in the order the firmware uses:
"""
import json
import logging
import os
import re

_LOG = logging.getLogger("ring")

OCCASIONS = ["listening", "thinking", "replying", "error", "incoming_call", "baby_cry", "alarm"]
EFFECTS = ["solid", "breathing", "rotating"]
_HEX = re.compile(r"^#?[0-9a-fA-F]{6}$")


def node_of(device_entities: list[dict], states: dict) -> str | None:
    """The board's node name (its «Node name» diagnostic sensor), or None if it has no LED ring support."""
    for ent in device_entities:
        eid = ent["entity_id"]
        if eid.startswith("sensor.") and (eid.endswith("_nome_nodo") or eid.endswith("_node_name")):
            v = (states.get(eid) or {}).get("state")
            if v and v not in ("unknown", "unavailable"):
                return v
    return None


def stock_colour(device_entities: list[dict], states: dict) -> str:
    """Colour of the board's «Voice Assistant Ring» light: the colour of the firmware's own animations."""
    for ent in device_entities:
        eid = ent["entity_id"]
        if eid.startswith("light.") and eid.endswith("_voice_assistant_ring"):
            rgb = ((states.get(eid) or {}).get("attributes") or {}).get("rgb_color")
            if rgb and len(rgb) == 3:
                return "#%02x%02x%02x" % tuple(int(v) for v in rgb)
    return "#ffffff"


def speed_code(speed: float | None) -> int:
    """Speed factor (0.25x..4x) to the firmware's 6-bit code: 0 = default, else 32 + 16*log2(speed)."""
    import math
    if not speed or abs(speed - 1.0) < 1e-3:
        return 0
    return max(1, min(63, round(32 + 16 * math.log2(max(0.25, min(4.0, float(speed)))))))


def colors_hex(colors: list[str]) -> str:
    if len(colors) != 12 or not all(_HEX.match(c or "") for c in colors):
        raise ValueError("12 colours #rrggbb are needed")
    return "".join(c.lstrip("#").lower() for c in colors)


class RingStore:
    def __init__(self, data_dir: str) -> None:
        self.path = os.path.join(data_dir, "ring.json")
        try:
            with open(self.path) as f:
                self.designs: dict = json.load(f)
        except (OSError, ValueError):
            self.designs = {}

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.designs, f, indent=1)
        os.replace(tmp, self.path)

    def get(self, device_id: str) -> dict:
        return self.designs.get(device_id, {})

    def put(self, device_id: str, occasion: str, colors: list[str], effect: str, speed: float = 1.0) -> None:
        self.designs.setdefault(device_id, {})[occasion] = {"colors": colors, "effect": effect, "speed": speed}
        self._save()

    def remove(self, device_id: str, occasion: str) -> None:
        self.designs.get(device_id, {}).pop(occasion, None)
        self._save()

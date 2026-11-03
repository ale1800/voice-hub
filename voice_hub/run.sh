#!/bin/sh
# Turns the add-on options into command-line arguments.
# Data (enrolled people, collected commands) lives in /share/voice_hub so it
# can be reached from the Samba / SSH add-ons and survives reinstalls.
set -e
exec python3 - <<'PY'
import json, os, sys
o = json.load(open("/data/options.json"))
args = ["python3", "/app/server.py",
        "--stt-entity", o["stt_entity"], "--language", o.get("language") or "",
        "--model", os.environ["VOICE_HUB_MODEL"], "--data", os.environ["VOICE_HUB_DATA"],
        "--threshold", str(o["threshold"]), "--margin", str(o["margin"]),
        "--min-duration", str(o["min_duration"]), "--threads", str(o.get("threads", 2))]
if not o.get("capture", True): args.append("--no-capture")
if o.get("wyoming"): args.append("--wyoming")
if o.get("wake", True): args += ["--wake", "--wake-threshold", str(o.get("wake_threshold", 0.1))]
if o.get("save_rejected", True): args.append("--save-rejected")
if o.get("agent", True): args.append("--agent")
if o.get("collect"): args.append("--collect")
if o.get("debug"): args.append("--debug")
os.execvp(args[0], args)
PY

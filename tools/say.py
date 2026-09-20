#!/usr/bin/env python3
"""Inject a typed host line into the running director (test input, labelled as such).
  python tools/say.py "Please welcome Barack to the stage."
  python tools/say.py --server http://127.0.0.1:8704 "Over to Joe."   (or CUE_SERVER_HTTP=...)
"""
import json
import sys
import urllib.request

import os

args = sys.argv[1:]
server = os.environ.get("CUE_SERVER_HTTP", "http://127.0.0.1:8000")
if len(args) >= 2 and args[0] == "--server":
    server, args = args[1].rstrip("/"), args[2:]
text = " ".join(args).strip()
if not text:
    print(__doc__)
    sys.exit(1)
req = urllib.request.Request(f"{server}/api/say", data=json.dumps({"text": text}).encode(), headers={"content-type": "application/json"})
with urllib.request.urlopen(req, timeout=15) as r:
    print(json.dumps(json.load(r), indent=2))

#!/usr/bin/env python3
"""Verify the two provider keys in 10 seconds. Never prints a key.
  .venv/bin/python tools/smoke_keys.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import load_settings  # noqa: E402
from server.semantics import CueLLM  # noqa: E402


async def main() -> int:
    s = load_settings()
    ok = True
    from server.config import KEY_SOURCES
    llm_state = ('set (' + KEY_SOURCES.get('OPENAI_API_KEY', '?') + ')' if s.has_openai
                 else (f'not set, local LLM at {s.llm_base_url}' if s.llm_base_url else 'MISSING -> degraded: rules-only interpreter, no VLM'))
    print(f"OPENAI_API_KEY   : {llm_state}  (model {s.llm_model})")
    print(f"DEEPGRAM_API_KEY : {'set (' + KEY_SOURCES.get('DEEPGRAM_API_KEY', '?') + ')' if s.has_deepgram else 'MISSING -> degraded: no live transcription'}  (model {s.deepgram_model})")
    missing = not (s.has_llm and s.has_deepgram)
    if s.llm_base_url:
        import httpx
        try:
            r = httpx.get(s.llm_base_url + "/models", timeout=5)
            names = [m.get("id") for m in r.json().get("data", [])]
            print(f"Local LLM server: HTTP {r.status_code}, models {names}; {'model present' if s.llm_model in names else 'MODEL MISSING: ollama pull ' + s.llm_model}")
            ok = ok and r.status_code == 200 and s.llm_model in names
        except Exception as e:
            ok = False
            print(f"Local LLM server FAILED: {type(e).__name__}: {e} (run: ollama serve)")
    if s.has_openai:
        from openai import AsyncOpenAI
        c = AsyncOpenAI()
        t0 = time.perf_counter()
        try:
            r = await asyncio.wait_for(c.responses.parse(
                model=s.llm_model,
                input=[{"role": "system", "content": "Roster: sarah (Sarah Tan). Return meaning only."},
                       {"role": "user", "content": "Latest clause: Please welcome Sarah."}],
                text_format=CueLLM, max_output_tokens=200), timeout=20)
            print(f"OpenAI structured output OK in {(time.perf_counter() - t0) * 1000:.0f} ms -> {r.output_parsed}")
        except Exception as e:
            ok = False
            print(f"OpenAI FAILED: {type(e).__name__}: {e}")
    if s.has_deepgram:
        import httpx
        try:
            r = httpx.get("https://api.deepgram.com/v1/projects", headers={"Authorization": f"Token {s.deepgram_api_key}"}, timeout=10)
            print(f"Deepgram auth {'OK' if r.status_code == 200 else 'FAILED'} (HTTP {r.status_code})")
            ok = ok and r.status_code == 200
        except Exception as e:
            ok = False
            print(f"Deepgram FAILED: {type(e).__name__}: {e}")
    try:
        from server.speech import list_input_devices
        devs = list_input_devices()
        print("Mic inputs:", ", ".join(f"[{d['index']}] {d['name']}" for d in devs) or "none found")
    except Exception as e:
        print(f"Mic listing failed: {e}")
    if missing:
        print("exit 2: at least one key is missing (the show still runs, degraded)")
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

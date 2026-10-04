#!/usr/bin/env python3
"""Evaluate the LLM interpreter on the team's corpora (accuracy and latency).

  .venv/bin/python tools/eval_llm.py --models qwen2.5:3b llama3.2:3b            # local Ollama via CUE_LLM_BASE_URL
  .venv/bin/python tools/eval_llm.py --models gpt-4.1-mini --set heldout         # OpenAI when a key is present

The corpora (scripts/data/) and the roster (apps/api/src/cue_api/semantics/roster.json) live in the
team repo; clone it next to this one first:
  git clone https://github.com/chocoHacks33/CUE.git ../hackmit_2026_cue

Uses the same LLMParser + validate() as the live server. A case passes when the action is in the
allowed set and, for SHOW, the target matches. Wrong cuts (a confident SHOW/WIDE/HOST where the
corpus says HOLD, or SHOW of the wrong person) are listed: those are the ones that matter.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import load_settings  # noqa: E402
from server.semantics import Action, Cue, LLMParser, Person, Roster, RuleParser, guard_llm_cue, validate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent / "hackmit_2026_cue"


def load_cases(name: str):
    raw = json.loads((REPO / "scripts" / "data" / f"{name}.json").read_text())
    return raw["cases"] if isinstance(raw, dict) else raw


def pct(xs, p):
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(round(p / 100 * (len(ys) - 1))))] if ys else None


a_no_guard = False


async def run(model: str, set_name: str, s, roster: Roster, concurrency: int, repeat: int) -> dict:
    parser = LLMParser(model, roster, "", timeout_s=max(s.llm_timeout_s, 30), base_url=s.llm_base_url, api_key=s.llm_api_key)
    rules = RuleParser(roster)
    ids = roster.ids
    cases = [c for c in load_cases(set_name) if set(c.get("target_guest_ids") or []) <= ids and c.get("cat") != "DUPLICATE_NAME"] * repeat
    sem = asyncio.Semaphore(concurrency)
    lat, rows = [], []

    async def one(c):
        async with sem:
            try:
                parsed, ms = await parser.parse(c["say"])
                cue = Cue(list(parsed.target_ids), parsed.scope, parsed.intent, parsed.temporal_intent, parsed.action, parsed.evidence_text)
                cue = guard_llm_cue(cue, c["say"], rules, roster) if not a_no_guard else validate(cue, roster)
                err = ""
            except Exception as e:
                cue, ms, err = None, 0.0, f"{type(e).__name__}: {e}"[:120]
            lat.append(ms)
            allowed = set(c["action"]); tgt = c.get("target_guest_ids") or []
            got = cue.action.value if cue else "ERROR"
            ok = cue is not None and got in allowed and (got != "SHOW" or not tgt or cue.target_ids == tgt)
            wrong_cut = cue is not None and ((got not in allowed and got != "HOLD") or (got == "SHOW" and tgt and cue.target_ids != tgt))
            rows.append({"id": c["id"], "cat": c.get("cat"), "say": c["say"], "expected": sorted(allowed), "expected_targets": tgt,
                         "got": got, "targets": cue.target_ids if cue else None, "ok": ok, "wrong_cut": wrong_cut, "ms": round(ms), "err": err})

    t0 = time.perf_counter()
    await asyncio.gather(*(one(c) for c in cases))
    wall = time.perf_counter() - t0
    rows.sort(key=lambda r: r["id"])
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["cat"], [0, 0]); by_cat[r["cat"]][1] += 1; by_cat[r["cat"]][0] += r["ok"]
    return {"model": model, "set": set_name, "n": len(rows), "passed": sum(r["ok"] for r in rows), "wrong_cuts": [r for r in rows if r["wrong_cut"]],
            "errors": sum(1 for r in rows if r["err"]), "latency_ms": {"p50": round(statistics.median(lat)), "p95": round(pct(lat, 95)), "max": round(max(lat))},
            "wall_s": round(wall, 1), "by_cat": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())}, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--set", dest="set_name", choices=["adversarial", "heldout", "both"], default="adversarial")
    ap.add_argument("--concurrency", type=int, default=1, help="1 = serial (matches live latency); higher = faster eval")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-guard", action="store_true", help="raw model output, without the deterministic guard the live server applies")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "reports"))
    a = ap.parse_args()
    global a_no_guard
    a_no_guard = a.no_guard
    if not (REPO / "scripts" / "data").is_dir():
        print(f"team repo not found at {REPO}: git clone https://github.com/chocoHacks33/CUE.git {REPO}")
        return 1
    s = load_settings()
    if not s.has_llm:
        print("no LLM configured: set CUE_LLM_BASE_URL (+ CUE_LLM_MODEL) or OPENAI_API_KEY"); return 1
    guests = json.loads((REPO / "apps/api/src/cue_api/semantics/roster.json").read_text())["guests"]
    roster = Roster([Person(g["id"], g["name"], g["aliases"], g["role"]) for g in guests])
    sets = ["adversarial", "heldout"] if a.set_name == "both" else [a.set_name]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = []
    for model in a.models:
        for set_name in sets:
            r = asyncio.run(run(model, set_name, s, roster, a.concurrency, a.repeat))
            acc = 100 * r["passed"] / r["n"]
            print(f"\n{model} on {set_name}: {r['passed']}/{r['n']} = {acc:.0f}%  wrong cuts {len(r['wrong_cuts'])}  errors {r['errors']}  "
                  f"latency p50 {r['latency_ms']['p50']} ms p95 {r['latency_ms']['p95']} ms  ({r['wall_s']}s wall)")
            print("   by category:", r["by_cat"])
            for w in r["wrong_cuts"]:
                print(f"   WRONG CUT {w['id']}: {w['say']!r} -> {w['got']} {w['targets']} (expected {w['expected']} {w['expected_targets']})")
            for x in [x for x in r["rows"] if not x["ok"] and not x["wrong_cut"]][:8]:
                print(f"   miss      {x['id']}: {x['say']!r} -> {x['got']} {x['targets']} (expected {x['expected']}) {x['err']}")
            ts = time.strftime("%Y%m%d_%H%M%S")
            (out / f"llm_eval_{set_name}_{model.replace('/', '_').replace(':', '-')}_{ts}.json").write_text(json.dumps(r, indent=2))
            summary.append((model, set_name, acc, len(r["wrong_cuts"]), r["latency_ms"]))
    print("\n| model | set | accuracy | wrong cuts | p50 ms | p95 ms |\n|---|---|---|---|---|---|")
    for m, sn, acc, wc, lat in summary:
        print(f"| {m} | {sn} | {acc:.0f}% | {wc} | {lat['p50']} | {lat['p95']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())

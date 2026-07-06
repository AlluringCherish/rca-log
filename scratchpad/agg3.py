#!/usr/bin/env python3
"""Aggregate the 3-condition d-run (W/O unit / NL 20k / KV 20k) into one table.
Reads predictions.json (progress), evaluation.json (accuracy), traces (timing)."""
import json, glob, os, sys

BASE = "/data/rca-proto/output"
CONDS = [
    ("full90d_nounit", "W/O unit"),
    ("full90d_nl", "NL 20k"),
    ("full90d_kv", "KV 20k"),
]

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def load(run):
    d = os.path.join(BASE, run)
    n = 0
    pj = os.path.join(d, "predictions.json")
    if os.path.exists(pj):
        try:
            n = len(json.load(open(pj)))
        except Exception:
            n = 0
    ov = {}
    ej = os.path.join(d, "evaluation.json")
    if os.path.exists(ej):
        try:
            ov = json.load(open(ej)).get("summary", {}).get("overall", {})
        except Exception:
            ov = {}
    ttft, dec, steps, hits, calls = [], [], [], 0, 0
    for f in glob.glob(os.path.join(d, "traces", "*.json")):
        try:
            t = json.load(open(f))
        except Exception:
            continue
        ti = t.get("timing_s") or {}
        nc = ti.get("analyst_n_calls") or 0
        if nc:
            ttft.append(ti.get("analyst_prefill_s", 0) / nc * 1000)
            dec.append(ti.get("analyst_decode_s", 0) / nc * 1000)
            hits += ti.get("analyst_cache_hits", 0) or 0
            calls += nc
        steps.append(len(t.get("steps", [])))
    return dict(n=n, ov=ov,
                ttft=mean(ttft), dec=mean(dec),
                steps=mean(steps),
                hit=(100.0 * hits / calls if calls else 0.0))

rows = [(lbl, run, load(run)) for run, lbl in CONDS]

hdr = f"{'cond':9} | {'done':>5} | {'cmp@1':>5} | {'rsn@1':>5} | {'both@1':>6} | {'rsn@3':>5} | {'both@3':>6} | {'TTFT':>6} | {'dec':>6} | {'/call':>6} | {'stp':>4} | {'hit%':>5} | {'sec/cs':>6}"
print(hdr)
print("-" * len(hdr))
for lbl, run, r in rows:
    ov = r["ov"]
    def g(k):
        v = ov.get(k)
        return f"{v:.2f}" if isinstance(v, (int, float)) else "  - "
    percall = (r["ttft"] + r["dec"])
    tsm = ov.get("time_s_mean")
    tsm = f"{tsm:.1f}" if isinstance(tsm, (int, float)) else "  - "
    print(f"{lbl:9} | {r['n']:>3}/90 | {g('top_1_component'):>5} | {g('top_1_reason'):>5} | "
          f"{g('top_1_both'):>6} | {g('top_3_reason'):>5} | {g('top_3_both'):>6} | "
          f"{r['ttft']:>6.0f} | {r['dec']:>6.0f} | {percall:>6.0f} | {r['steps']:>4.1f} | {r['hit']:>5.0f} | {tsm:>6}")
print("\n(TTFT/dec//call in ms per analyst-call; sec/cs = mean wall-clock per case;"
      " cmp/rsn/both = component/reason/both accuracy; metrics over completed cases only)")

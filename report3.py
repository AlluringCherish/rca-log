#!/usr/bin/env python3
"""3-condition comparison table: accuracy (top-1/2/3) + TTFT/decode/exec/steps.

Usage:
  python3 report3.py --nounit full90_nounit --nl full90_unit_nl --kv full90_unit_kv
"""

import argparse
import json
import os
from statistics import mean

from benchmark.re2_ob import split_rank


def load(run, output_dir="output"):
    return json.load(open(os.path.join(output_dir, run, "predictions.json")))


def stats(rows):
    n = len(rows)
    m = {"n": n}
    for k in (1, 2, 3):
        m[f"c{k}"] = sum(split_rank(r["answer"])[0] in [split_rank(p)[0] for p in r["prediction"][:k]] for r in rows) / n
        m[f"r{k}"] = sum(split_rank(r["answer"])[1] in [split_rank(p)[1] for p in r["prediction"][:k]] for r in rows) / n
        m[f"b{k}"] = sum(split_rank(r["answer"]) in [split_rank(p) for p in r["prediction"][:k]] for r in rows) / n
    # timing: TTFT/decode are per-ANALYST-call means (ms); exec/steps per case
    a_pre = a_dec = a_calls = 0.0
    execs, steps = [], []
    for r in rows:
        t = r.get("timing_s") or {}
        a_pre += t.get("analyst_prefill_s", 0.0)
        a_dec += t.get("analyst_decode_s", 0.0)
        a_calls += t.get("analyst_n_calls", 0)
        if t.get("all") is not None:
            execs.append(float(t["all"]))
        steps.append(len(r.get("steps", [])))
    d = max(a_calls, 1)
    m["ttft_ms"] = 1000 * a_pre / d
    m["decode_ms"] = 1000 * a_dec / d
    m["exec_s"] = mean(execs) if execs else 0.0
    m["steps"] = mean(steps) if steps else 0.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nounit", required=True)
    ap.add_argument("--nl", required=True)
    ap.add_argument("--kv", required=True)
    ap.add_argument("--output-dir", default="output")
    args = ap.parse_args()

    conds = [
        ("W/O reasoning unit", args.nounit),
        ("W/ unit (natural lang)", args.nl),
        ("W/ unit (kv-cache)", args.kv),
    ]
    data = {label: stats(load(run, args.output_dir)) for label, run in conds}
    n = next(iter(data.values()))["n"]

    print(f"\n{'='*104}")
    print(f"3-CONDITION COMPARISON (n={n} cases each)   "
          f"[both = component AND reason correct; TTFT/decode = ms per analyst call; exec/steps per case]")
    print("=" * 104)
    hdr = (f"{'condition':24} | {'both@1':>6} {'both@2':>6} {'both@3':>6} | "
           f"{'comp@1':>6} {'reason@1':>8} | {'TTFT':>7} {'decode':>7} {'exec':>7} {'steps':>6}")
    print(hdr)
    print("-" * len(hdr))
    for label, _ in conds:
        m = data[label]
        print(f"{label:24} | {m['b1']:>6.2f} {m['b2']:>6.2f} {m['b3']:>6.2f} | "
              f"{m['c1']:>6.2f} {m['r1']:>8.2f} | "
              f"{m['ttft_ms']:>6.0f}m {m['decode_ms']:>6.0f}m {m['exec_s']:>6.1f}s {m['steps']:>6.1f}")
    print("-" * len(hdr))

    # deltas of interest
    nl, kv, no = data["W/ unit (natural lang)"], data["W/ unit (kv-cache)"], data["W/O reasoning unit"]
    print(f"\nunit effect  (NL vs W/O):   both@1 {nl['b1']-no['b1']:+.2f}   both@3 {nl['b3']-no['b3']:+.2f}   "
          f"exec {nl['exec_s']-no['exec_s']:+.1f}s")
    print(f"kv-cache effect (KV vs NL): both@1 {kv['b1']-nl['b1']:+.2f}   "
          f"TTFT {kv['ttft_ms']-nl['ttft_ms']:+.0f}ms   decode {kv['decode_ms']-nl['decode_ms']:+.0f}ms   "
          f"exec {kv['exec_s']-nl['exec_s']:+.1f}s")


if __name__ == "__main__":
    main()

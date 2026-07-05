#!/usr/bin/env python3
"""Compare KV-prefix-cache runs: accuracy (top-1/2/3) + Analyst timing breakdown.

For each run: top-1/2/3 component/reason/both accuracy, and per-Analyst-call means
for TTFT (prefill_s), decode_s, cache build_s, effective prefill (prefill+build),
gen tokens, cache hit rate, plus per-case wall time.

Usage:
  python3 report_kv.py --runs full90_prefix_off,full90_prefix_on --legacy full90_with
"""

import argparse
import json
import os

from benchmark.re2_ob import split_rank


def load(run, output_dir="output"):
    rows = json.load(open(os.path.join(output_dir, run, "predictions.json")))
    return rows


def accuracy(rows):
    n = len(rows)
    m = {}
    for k in (1, 2, 3):
        m[f"c{k}"] = sum(split_rank(r["answer"])[0] in [split_rank(p)[0] for p in r["prediction"][:k]] for r in rows) / n
        m[f"r{k}"] = sum(split_rank(r["answer"])[1] in [split_rank(p)[1] for p in r["prediction"][:k]] for r in rows) / n
        m[f"b{k}"] = sum(split_rank(r["answer"]) in [split_rank(p) for p in r["prediction"][:k]] for r in rows) / n
    m["n"] = n
    return m


def timing(rows, role="analyst"):
    """Mean-per-call (across all cases) of the role's timing metrics."""
    calls = build = prefill = decode = total = gen = hits = 0
    wall = []
    for r in rows:
        t = r.get("timing_s") or {}
        c = t.get(f"{role}_n_calls", 0)
        calls += c
        hits += t.get(f"{role}_cache_hits", 0)
        prefill += t.get(f"{role}_prefill_s", 0.0)
        decode += t.get(f"{role}_decode_s", 0.0)
        build += t.get(f"{role}_cache_build_s", 0.0)
        total += t.get(f"{role}_total_s", 0.0)
        gen += t.get(f"{role}_gen_tokens", 0)
        if t.get("all") is not None:
            wall.append(float(t["all"]))
    d = max(calls, 1)
    return {
        "calls": calls,
        "hit_rate": hits / d,
        "ttft_ms": 1000 * prefill / d,
        "build_ms": 1000 * build / d,
        "eff_prefill_ms": 1000 * (prefill + build) / d,
        "decode_ms": 1000 * decode / d,
        "gen_tok": gen / d,
        "case_wall_s": (sum(wall) / len(wall)) if wall else 0.0,
        "wall_total_s": sum(wall),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run ids to compare")
    ap.add_argument("--legacy", default=None, help="reference run id (accuracy only)")
    ap.add_argument("--output-dir", default="output")
    args = ap.parse_args()

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    all_runs = runs + ([args.legacy] if args.legacy else [])
    data = {r: load(r, args.output_dir) for r in all_runs}

    print("=" * 92)
    print("ACCURACY (top-1/2/3)")
    print("=" * 92)
    print(f"{'run':22} {'n':>3} | {'comp@1/2/3':>16} | {'reason@1/2/3':>16} | {'both@1/2/3':>16}")
    for r in all_runs:
        a = accuracy(data[r])
        print(f"{r:22} {a['n']:>3} | "
              f"{a['c1']:.2f} {a['c2']:.2f} {a['c3']:.2f}    | "
              f"{a['r1']:.2f} {a['r2']:.2f} {a['r3']:.2f}    | "
              f"{a['b1']:.2f} {a['b2']:.2f} {a['b3']:.2f}")

    print("\n" + "=" * 92)
    print("ANALYST timing (mean per call, ms) — TTFT = generate prefill; eff = TTFT + amortized cache build")
    print("=" * 92)
    print(f"{'run':22} {'calls':>5} {'hit%':>5} {'TTFT':>7} {'build':>6} {'eff_pre':>8} {'decode':>7} {'gen_tok':>7} {'case_s':>7}")
    for r in runs:
        t = timing(data[r], "analyst")
        print(f"{r:22} {t['calls']:>5} {100*t['hit_rate']:>4.0f}% {t['ttft_ms']:>6.0f} {t['build_ms']:>6.0f} "
              f"{t['eff_prefill_ms']:>8.0f} {t['decode_ms']:>7.0f} {t['gen_tok']:>7.0f} {t['case_wall_s']:>7.1f}")

    if len(runs) == 2:
        a, b = timing(data[runs[0]], "analyst"), timing(data[runs[1]], "analyst")
        print(f"\nΔ ({runs[1]} vs {runs[0]}):  TTFT {b['ttft_ms']-a['ttft_ms']:+.0f}ms  "
              f"eff_prefill {b['eff_prefill_ms']-a['eff_prefill_ms']:+.0f}ms  "
              f"decode {b['decode_ms']-a['decode_ms']:+.0f}ms  case_wall {b['case_wall_s']-a['case_wall_s']:+.1f}s")


if __name__ == "__main__":
    main()

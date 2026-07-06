#!/usr/bin/env python3
"""Recompute RCA metrics directly from a run's per-case trace files.

Robust to predictions.json being overwritten/partial (traces are per-case and
persist). Replicates benchmark/re2_ob.evaluate_cases metric definitions:
top-k component/reason/both over the unique-preserving prediction order.
"""
import glob
import json
import os
from statistics import mean

BASE = "/data/rca-proto/output"


def _uniq(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def split_rank(rank):
    if "_" not in rank:
        return rank, "unknown"
    c, s = rank.split("_", 1)
    return c, s


def eval_traces(run):
    """run: output-dir name (e.g. 'full90e_unit_kv'). Returns metrics dict or None."""
    d = os.path.join(BASE, run, "traces")
    traces = []
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            traces.append(json.load(open(f)))
        except Exception:
            pass
    if not traces:
        return None

    acc = {f"top_{k}_{m}": [] for k in range(1, 4) for m in ("component", "reason", "both")}
    execs, steps, ttft, dec, gtok = [], [], [], [], []
    in_tok, out_tok = [], []   # per-case analyst input / output token totals
    hits = calls = 0
    for t in traces:
        ans = t.get("answer", "")
        pred = _uniq([p for p in (t.get("prediction") or []) if p])[:5]
        comps = _uniq([split_rank(p)[0] for p in pred])
        reasons = _uniq([split_rank(p)[1] for p in pred])
        fine = [split_rank(p) for p in pred]
        ac, ar = split_rank(ans)
        for k in range(1, 4):
            acc[f"top_{k}_component"].append(ac in comps[:k])
            acc[f"top_{k}_reason"].append(ar in reasons[:k])
            acc[f"top_{k}_both"].append((ac, ar) in fine[:k])
        ti = t.get("timing_s") or {}
        if ti.get("all") is not None:
            execs.append(ti["all"])
        steps.append(len(t.get("steps", [])))
        # per-case analyst token totals (summed across the case's analyst calls)
        pfx = ti.get("analyst_prefix_tokens", 0) or 0
        sfx = ti.get("analyst_suffix_tokens", 0) or 0
        gen = ti.get("analyst_gen_tokens", 0) or 0
        in_tok.append(pfx + sfx)
        out_tok.append(gen)
        nc = ti.get("analyst_n_calls") or 0
        if nc:
            ttft.append(ti.get("analyst_prefill_s", 0) / nc * 1000)
            dec.append(ti.get("analyst_decode_s", 0) / nc * 1000)
            gtok.append(gen / nc)
            hits += ti.get("analyst_cache_hits", 0) or 0
            calls += nc

    out = {"n": len(traces)}
    for k, v in acc.items():
        out[k] = sum(v) / len(v)
    out["exec_s"] = mean(execs) if execs else 0.0
    out["steps"] = mean(steps) if steps else 0.0
    out["ttft_ms"] = mean(ttft) if ttft else 0.0
    out["dec_ms"] = mean(dec) if dec else 0.0
    out["percall_ms"] = out["ttft_ms"] + out["dec_ms"]
    out["tok_call"] = mean(gtok) if gtok else 0.0
    out["in_tok"] = mean(in_tok) if in_tok else 0.0      # analyst input tokens / case
    out["out_tok"] = mean(out_tok) if out_tok else 0.0   # analyst output tokens / case
    out["tot_tok"] = out["in_tok"] + out["out_tok"]      # total tokens / case
    out["hit_pct"] = 100.0 * hits / calls if calls else 0.0
    return out


if __name__ == "__main__":
    import sys
    for run in sys.argv[1:]:
        r = eval_traces(run)
        print(run, json.dumps(r, indent=None) if r else "(no traces)")

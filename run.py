#!/usr/bin/env python3
"""Run the event-based RCA agent over preprocessed RE2-OB cases.

Defaults to the local Qwen3 backend. Writes per-case traces plus predictions.json
and evaluation.json (top-1/2/3 component / reason / both).

Usage:
  python3 run.py --cases problem_000001 --max-steps 6 --verbose
  python3 run.py --case-limit 6 --llm-backend local --local-model /data/models/Qwen3-8B
  OPENAI_API_KEY=... OPENAI_MODEL=gpt-4o-mini python3 run.py --llm-backend openai --parallelism 4
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from agent.analyst import Analyst
from agent.controller import Controller
from agent.loop import run_case
from benchmark.re2_ob import discover_event_cases, evaluate_cases, write_outputs
from common.llm import LLMClient
from reasoning.units import UnitDB, load_units


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events-root", default="data/re2-ob/events")
    p.add_argument("--cases", default=None, help="Comma-separated problem ids or source names.")
    p.add_argument("--case-limit", type=int, default=None)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--run-id", default=None, help="Subdirectory for this run's outputs.")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--units-path", default="reasoning/seed_units.json")
    p.add_argument("--no-units", action="store_true",
                   help="Ablation: disable reasoning units (Analyst gets no reasoning unit).")
    p.add_argument("--kv-prefix-cache", action="store_true",
                   help="Cache the Analyst prompt prefix (system+unit) as KV; recompute only the dynamic suffix (local backend only).")
    p.add_argument("--kv-offload-cpu", action="store_true",
                   help="Store prefix KV on host RAM and copy to GPU per call (needed when large prefixes x units exceed VRAM).")
    p.add_argument("--big-prefix", metavar="PATH", nargs="?", const="reasoning/rca_reference.txt", default=None,
                   help="Append a large fixed RCA reference to the Analyst system prompt (grows the cached prefix to ~20k). Optional path; default reasoning/rca_reference.txt.")
    p.add_argument("--llm-backend", choices=["local", "openai"], default="local")
    p.add_argument("--model", default=None, help="OpenAI/OpenRouter model id.")
    p.add_argument("--local-model", default="/data/models/Qwen3-8B")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--cpu", action="store_true", help="Run the local model on CPU.")
    p.add_argument("--parallelism", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, run_id)
    trace_dir = os.path.join(out_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)

    cases = args.cases.split(",") if args.cases else None
    discovered = discover_event_cases(args.events_root, cases=cases, case_limit=args.case_limit)
    print(f"Discovered {len(discovered)} case(s). Output -> {out_dir}", file=sys.stderr)

    units = [] if args.no_units else load_units(args.units_path)
    unit_db = UnitDB(units)
    print(f"Reasoning units: {'DISABLED (ablation)' if args.no_units else f'{len(units)} loaded (single-unit select)'}",
          file=sys.stderr)
    if args.kv_prefix_cache and args.llm_backend != "local":
        print("--kv-prefix-cache is only supported by the local backend; ignoring.", file=sys.stderr)
    llm = LLMClient(
        model=args.model,
        temperature=args.temperature,
        backend=args.llm_backend,
        local_model=args.local_model,
        max_new_tokens=args.max_new_tokens,
        cpu=args.cpu,
        enable_thinking=args.enable_thinking,
        kv_prefix_cache=args.kv_prefix_cache,
        kv_offload_cpu=args.kv_offload_cpu,
    )
    llm.prepare()
    if args.llm_backend == "local":
        print(f"Prefix KV-cache: {'ON' if args.kv_prefix_cache else 'OFF'}", file=sys.stderr)
    controller = Controller(llm)
    if args.big_prefix:
        from common.prompts import ANALYST_SYSTEM_PROMPT
        with open(args.big_prefix, encoding="utf-8") as f:
            reference = f.read()
        analyst = Analyst(llm, system_prompt=ANALYST_SYSTEM_PROMPT + "\n\n" + reference)
        print(f"Big prefix: appended {args.big_prefix} to Analyst system prompt.", file=sys.stderr)
    else:
        analyst = Analyst(llm)

    # Pre-store each reasoning unit's prefix KV once ("미리 저장") so inference reuses it.
    if args.kv_prefix_cache and args.llm_backend == "local" and not args.no_units:
        warm_msgs = [analyst.prefix_messages(up) for up in unit_db.all_unit_prompts()]
        info = llm.warm_prefixes(warm_msgs)
        print(f"Warmed {info.get('built', 0)} unit prefixes in {info.get('total_s', 0):.1f}s "
              f"(registry={info.get('registry_size', 0)})", file=sys.stderr)

    parallelism = args.parallelism
    if args.llm_backend == "local" and parallelism != 1:
        print("Local backend is single-GPU; forcing --parallelism 1.", file=sys.stderr)
        parallelism = 1

    def process(case) -> Dict[str, Any]:
        print(f"[{case.case_id}] start", file=sys.stderr)
        record = run_case(case, controller, analyst, unit_db,
                          max_steps=args.max_steps, verbose=args.verbose)
        _write_trace(trace_dir, record)
        pred = record["prediction"]
        ans = case.answer_rank
        hit = "HIT" if ans in pred[:3] else "miss"
        print(f"[{case.case_id}] done ({hit}) answer={ans} pred={pred[:3]} "
              f"err={record.get('error')}", file=sys.stderr)
        return record

    records: List[Dict[str, Any]] = []
    if parallelism > 1:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            for record in pool.map(process, discovered):
                records.append(record)
                write_outputs(out_dir, records)  # incremental
    else:
        for case in discovered:
            records.append(process(case))
            write_outputs(out_dir, records)  # incremental

    evaluation = evaluate_cases(records)
    _print_summary(evaluation, len(records))
    print(f"\nOutputs written to {out_dir}", file=sys.stderr)


def _write_trace(trace_dir: str, record: Dict[str, Any]) -> None:
    case = record["case"]
    payload = {
        "case": case.case_id,
        "answer": case.answer_rank,
        "prediction": record["prediction"],
        "final_ranking": record["final_ranking"],
        "error": record.get("error"),
        "timing_s": record.get("timing_s"),
        "steps": record.get("steps", []),
        "findings": record.get("findings"),
    }
    with open(os.path.join(trace_dir, f"{case.case_id}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _print_summary(evaluation: Dict[str, Any], n: int) -> None:
    summary = evaluation.get("summary", {}).get("overall", {})
    if not summary:
        return
    print(f"\n=== Evaluation over {n} case(s) ===")
    print(f"{'k':>3} | {'component':>9} | {'reason':>7} | {'both':>6}")
    for k in (1, 2, 3):
        print(f"{k:>3} | {summary.get(f'top_{k}_component', 0):>9.3f} | "
              f"{summary.get(f'top_{k}_reason', 0):>7.3f} | {summary.get(f'top_{k}_both', 0):>6.3f}")


if __name__ == "__main__":
    main()

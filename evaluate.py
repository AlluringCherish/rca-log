#!/usr/bin/env python3
"""Recompute evaluation metrics from a run's predictions or traces.

Usage:
  python evaluate.py --predictions output/<run_id>/predictions.json
  python evaluate.py --traces-dir output/<run_id>/traces --events-root data/events
"""

import argparse
import json
import os
from typing import Any, Dict, List

from benchmark.re2_ob import Case, discover_event_cases, evaluate_cases, load_answers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", default=None, help="Path to a run's predictions.json.")
    p.add_argument("--traces-dir", default=None, help="Path to a run's traces/ directory.")
    p.add_argument("--events-root", default="data/re2-ob/events")
    p.add_argument("--out", default=None, help="Write evaluation JSON here (default: stdout).")
    return p.parse_args()


def _case_index(events_root: str) -> Dict[str, Case]:
    return {c.case_id: c for c in discover_event_cases(events_root)}


def load_records_from_predictions(path: str, events_root: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    index = _case_index(events_root)
    records = []
    for row in rows:
        case = index.get(row["case"])
        if case is None:
            continue
        records.append({
            "case": case,
            "prediction": row.get("prediction", []),
            "final_ranking": row.get("final_ranking", []),
            "steps": row.get("steps", []),
            "timing_s": row.get("timing_s", {}),
        })
    return records


def load_records_from_traces(traces_dir: str, events_root: str) -> List[Dict[str, Any]]:
    index = _case_index(events_root)
    records = []
    for name in sorted(os.listdir(traces_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(traces_dir, name), "r", encoding="utf-8") as f:
            trace = json.load(f)
        case = index.get(trace.get("case"))
        if case is None:
            continue
        records.append({
            "case": case,
            "prediction": trace.get("prediction", []),
            "final_ranking": trace.get("final_ranking", []),
            "steps": trace.get("steps", []),
            "timing_s": trace.get("timing_s", {}),
        })
    return records


def print_report(evaluation: Dict[str, Any]) -> None:
    summary = evaluation.get("summary", {})
    order = ["overall"] + sorted(k for k in summary if k != "overall")
    header = f"{'group':>10} | {'n':>3} | " + " | ".join(
        f"c@{k} r@{k} b@{k}" for k in (1, 2, 3))
    print(header)
    print("-" * len(header))
    for group in order:
        s = summary.get(group)
        if not s:
            continue
        cells = []
        for k in (1, 2, 3):
            cells.append(f"{s.get(f'top_{k}_component',0):.2f} "
                         f"{s.get(f'top_{k}_reason',0):.2f} {s.get(f'top_{k}_both',0):.2f}")
        print(f"{group:>10} | {int(s.get('n',0)):>3} | " + " | ".join(cells))
    print("\nLegend: c=component, r=reason, b=both (component AND reason) at top-k")


def main() -> None:
    args = parse_args()
    if args.predictions:
        records = load_records_from_predictions(args.predictions, args.events_root)
    elif args.traces_dir:
        records = load_records_from_traces(args.traces_dir, args.events_root)
    else:
        raise SystemExit("Provide --predictions or --traces-dir.")
    if not records:
        raise SystemExit("No records loaded (case ids may not match the events root).")

    evaluation = evaluate_cases(records)
    print_report(evaluation)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

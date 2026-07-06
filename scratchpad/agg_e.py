#!/usr/bin/env python3
"""Report the e-run 3 conditions with the full column set the user asked for:
TTFT, decoding, execution time, steps, and top-1/2/3 (component, reason, both)."""
import sys
sys.path.insert(0, "/data/rca-proto/scratchpad")
from traceval import eval_traces

CONDS = [
    ("full90e_nounit", "[1] no unit"),
    ("full90e_unit_nl", "[2] unit NL"),
    ("full90e_unit_kv", "[3] unit KV"),
]
# allow overriding the run set from argv (e.g. smoke run-ids)
if len(sys.argv) > 1:
    CONDS = [(a, a) for a in sys.argv[1:]]

def cell(r, k):
    v = r.get(k)
    return f"{v:.2f}" if isinstance(v, (int, float)) else "  -"

# --- timing + token block ---
h1 = (f"{'cond':11} | {'done':>5} | {'TTFT':>6} | {'decode':>6} | {'/call':>6} | {'exec/cs':>7} | "
      f"{'steps':>5} | {'in_tok':>6} | {'out_tok':>7} | {'tot_tok':>7} | {'hit%':>4}")
print(h1); print("-" * len(h1))
rows = [(lbl, eval_traces(run)) for run, lbl in CONDS]
for lbl, r in rows:
    if not r:
        print(f"{lbl:11} | {'0/90':>5} |" + " " * 5 + "(no traces yet)")
        continue
    print(f"{lbl:11} | {str(r['n'])+'/90':>5} | {r['ttft_ms']:>6.0f} | {r['dec_ms']:>6.0f} | "
          f"{r['percall_ms']:>6.0f} | {r['exec_s']:>7.1f} | {r['steps']:>5.1f} | "
          f"{r['in_tok']:>6.0f} | {r['out_tok']:>7.0f} | {r['tot_tok']:>7.0f} | {r['hit_pct']:>4.0f}")

# --- accuracy block: top-1/2/3 x component/reason/both ---
print()
h2 = f"{'cond':11} | {'comp@1':>6} {'@2':>4} {'@3':>4} | {'rsn@1':>6} {'@2':>4} {'@3':>4} | {'both@1':>6} {'@2':>4} {'@3':>4}"
print(h2); print("-" * len(h2))
for lbl, r in rows:
    if not r:
        print(f"{lbl:11} | (no traces yet)"); continue
    print(f"{lbl:11} | "
          f"{cell(r,'top_1_component'):>6} {cell(r,'top_2_component'):>4} {cell(r,'top_3_component'):>4} | "
          f"{cell(r,'top_1_reason'):>6} {cell(r,'top_2_reason'):>4} {cell(r,'top_3_reason'):>4} | "
          f"{cell(r,'top_1_both'):>6} {cell(r,'top_2_both'):>4} {cell(r,'top_3_both'):>4}")
print("\n(TTFT/decode//call = ms per analyst-call; exec/cs = wall-clock s per case; "
      "in/out/tot_tok = analyst input/output/total tokens per case; metrics over completed cases only)")

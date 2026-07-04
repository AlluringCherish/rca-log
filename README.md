# rca-proto

Event-based **Controller / Analyst** root-cause-analysis agent for the
**RCA-Eval RE2-OB** benchmark (Google Online Boutique microservices).

Raw telemetry is preprocessed once into a unified **event stream** (DiagFusion-style
multimodal events). The **Controller** plans which events to fetch via a small tool
API; the **Analyst** interprets them against a **reasoning template** composed from
distilled *if–then reasoning units*, owns the Stop decision, and emits the top-3
`(component, reason)` root causes.

Self-contained: nothing here imports the original `/data/log` project. The raw
benchmark is read-only and external; only compact events are stored locally.

## Pipeline

```
RE2-OB raw CSVs ──preprocess.py──▶ data/re2-ob/events/problem_XXXXXX/{events.jsonl,meta.json}
                                    data/re2-ob/topology.json        (shared OB call graph)
                                    data/re2-ob/events/answers.json  (ground-truth key, eval only)

                 ┌──────────── run.py (per case) ────────────┐
                 │  Controller ──plan/fetch──▶ EventToolRuntime (5 tools over events.jsonl)
                 │      ▲                              │
                 │      │ data_requests               ▼ event lines
                 │  Analyst ◀──reasoning template── TemplateComposer (matches reasoning units)
                 │      │ stop + final top-3
                 └──────┼─────────────────────────────────────┘
                        ▼
                 predictions.json + evaluation.json (top-1/2/3 component / reason / both)
```

## Roles (the key design)

- **Controller** (`agent/controller.py`) — *planning only*. Chooses which event data to
  fetch, translates the Analyst's `data_requests` into tool calls, never analyzes,
  never stops, never ranks.
- **Analyst** (`agent/analyst.py`) — *interpretation only*. Cannot call tools; binds the
  reasoning template's `<placeholders>` from event lines, requests more data by
  event-pattern, **owns Stop**, and outputs the final top-3 with justifications.

## Events

Six event types, all with `{id, type, window:[start,end], service, attrs, line}`,
timestamps in **relative seconds from case start**, 30s windows. Metric events carry
a `kpi` (cpu/mem/diskio/socket/latency.p50/p90). Sort strength is derived on the fly
(`event_magnitude` = |z|), not stored:

| type | source | meaning |
|------|--------|---------|
| `metric_anomaly` | simple_metrics.csv | robust-z deviation of a service KPI (cpu/mem/diskio/socket/latency), runs merged, persistence-gated |
| `metric_summary` | simple_metrics.csv | per-window value + z of one service KPI (trajectory) |
| `span_slowdown` | traces.csv | caller>callee/op p99 of raw span `duration` rose vs baseline (incl. downstream wait) |
| `error_code` | traces.csv | caller>callee/op emitting non-OK status above baseline |
| `log_pattern` | logts.csv + cluster_info.json | per-template count deviation / new template |
| `call_edge` | traces.csv | caller>callee call & error volume (topology) |

**Unsupervised baseline.** Anomalies use a robust median/MAD baseline over `[0, 600)`s,
deliberately shorter than the dataset's 720s injection offset. Preprocessing **never reads
`inject_time.txt`**; a `--warmup` (default 120s) suppresses start-up ramp transients; a
persistence gate keeps sustained runs (RE2-OB faults last ~12 min). `meta.baseline_contamination`
audits residual false positives.

**Anti-leak.** Cases live in opaque `problem_XXXXXX` folders under `data/re2-ob/events/`;
ground truth is only in `answers.json` (eval). `Case.agent_context()` exposes no
service/fault/inject_time. The Online Boutique call graph is common across cases, so it
is stored once at `data/re2-ob/topology.json` rather than per case.

## Reasoning units

`reasoning/seed_units.json` holds manually-seeded if–then units: a `trigger` (event-pattern
IF) and `reasoning` (CoT THEN with `<variables>`). `TemplateComposer` matches triggers against
fetched events (candidate-scoped) and concatenates matched units into the analysis procedure
the Analyst instantiates. Seeds cover cpu/mem/diskio/socket localization, latency
propagation-vs-locality, error cascade, log confirmation, and a final reason reranker.

## Usage

Note: this environment has `python3` (no `python`).

```bash
# 1) Preprocess (raw benchmark -> events). Full set:
python3 preprocess.py --source-root /mnt/data/logjun/data/log/Benchmarks/RE2-OB \
        --output-root data/re2-ob/events --cases all
# or a subset by problem id or source name:
python3 preprocess.py --cases problem_000001,checkoutservice_socket_1 --overwrite

# 2) Run the agent (default backend = local Qwen3 on GPU; do NOT pass --cpu on GPU):
python3 run.py --cases problem_000001 --max-steps 6 --verbose
python3 run.py --case-limit 6 --llm-backend local --local-model /data/models/Qwen3-8B
# OpenAI-compatible:
OPENAI_API_KEY=... OPENAI_MODEL=gpt-4o-mini python3 run.py --llm-backend openai --parallelism 4

# 3) Re-score an existing run:
python3 evaluate.py --predictions output/<run_id>/predictions.json
```

Outputs land in `output/<run_id>/`: `predictions.json`, `evaluation.json`,
`predictions.csv`, and per-case `traces/<problem_id>.json`.

## Known data notes

- **Log level**: only 24/90 cases' `logs.csv` carry `cluster_id`, so `log_pattern.level`
  is `"unknown"` for the rest (template text / count / z are always present).
- **diskio columns**: only disk-fault cases expose the injured service's `diskio` column.
  Logic never branches on column presence — "no data → no events".
- **Reason ambiguity**: socket faults manifest partly as cpu, loss faults partly as
  mem/error; the component is clear, the reason is what the reasoning units disambiguate.

## Layout

```
preprocess.py  run.py  evaluate.py
common/    llm.py (OpenAI + local Qwen3, no KV-graft)  prompts.py
benchmark/ re2_ob.py (Case, discovery, evaluation, answer key)
events/    schema.py  extract.py  store.py  tools.py
reasoning/ units.py  seed_units.json
agent/     controller.py  analyst.py  loop.py
data/re2-ob/  topology.json + events/ (preprocess output; events/ gitignored)
```

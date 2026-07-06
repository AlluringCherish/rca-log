#!/bin/bash
# minimal-output smoke: 3 cases x 3 conditions ([1] cot no-unit / [2] min unit NL / [3] min unit KV)
cd /data/rca-proto
export LD_LIBRARY_PATH=/data/cuda-535-lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
UNITS=reasoning/seed_units_v1.json
C=problem_000001,problem_000016,problem_000046

echo "=== S1 [1] cot_nounit START $(date +%H:%M:%S) ==="
python3 run.py --cases $C --max-steps 8 --no-units --output-style cot \
  --run-id smoke_g_cot_nounit > output/smoke_g_cot_nounit.log 2>&1
echo "S1 rc=$? end $(date +%H:%M:%S) traces=$(ls output/smoke_g_cot_nounit/traces/ 2>/dev/null | wc -l)"

echo "=== S2 [2] min_nl START $(date +%H:%M:%S) ==="
python3 run.py --cases $C --max-steps 8 --units-path $UNITS --output-style minimal \
  --run-id smoke_g_min_nl > output/smoke_g_min_nl.log 2>&1
echo "S2 rc=$? end $(date +%H:%M:%S) traces=$(ls output/smoke_g_min_nl/traces/ 2>/dev/null | wc -l)"

echo "=== S3 [3] min_kv START $(date +%H:%M:%S) ==="
python3 run.py --cases $C --max-steps 8 --units-path $UNITS --kv-prefix-cache --output-style minimal \
  --run-id smoke_g_min_kv > output/smoke_g_min_kv.log 2>&1
echo "S3 rc=$? end $(date +%H:%M:%S) traces=$(ls output/smoke_g_min_kv/traces/ 2>/dev/null | wc -l)"
echo "=== SMOKE_G DONE $(date +%H:%M:%S) ==="

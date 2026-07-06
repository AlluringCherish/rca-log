#!/bin/bash
# full-90 x 3: [1] cot no-unit / [2] minimal unit NL / [3] minimal unit KV
cd /data/rca-proto
export LD_LIBRARY_PATH=/data/cuda-535-lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
UNITS=reasoning/seed_units_v1.json

echo "=== R1 [1] cot_nounit START $(date +%H:%M:%S) ==="
python3 run.py --max-steps 8 --no-units --output-style cot \
  --run-id full90g_cot_nounit > output/full90g_cot_nounit.log 2>&1
echo "R1 rc=$? end $(date +%H:%M:%S) traces=$(ls output/full90g_cot_nounit/traces/ 2>/dev/null | wc -l)"

echo "=== R2 [2] min_nl START $(date +%H:%M:%S) ==="
python3 run.py --max-steps 8 --units-path $UNITS --output-style minimal \
  --run-id full90g_min_nl > output/full90g_min_nl.log 2>&1
echo "R2 rc=$? end $(date +%H:%M:%S) traces=$(ls output/full90g_min_nl/traces/ 2>/dev/null | wc -l)"

echo "=== R3 [3] min_kv START $(date +%H:%M:%S) ==="
python3 run.py --max-steps 8 --units-path $UNITS --kv-prefix-cache --output-style minimal \
  --run-id full90g_min_kv > output/full90g_min_kv.log 2>&1
echo "R3 rc=$? end $(date +%H:%M:%S) traces=$(ls output/full90g_min_kv/traces/ 2>/dev/null | wc -l)"
echo "=== FULL90G ALL DONE $(date +%H:%M:%S) ==="

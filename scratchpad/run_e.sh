#!/bin/bash
# e-run: 3 conditions x 90 cases, original 9-unit set, standard output.
cd /data/rca-proto
export LD_LIBRARY_PATH=/data/cuda-535-lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
UNITS=reasoning/seed_units_v1.json

echo "=== R1 [1] no unit START $(date +%H:%M:%S) ==="
python3 run.py --max-steps 8 --no-units --run-id full90e_nounit > output/full90e_nounit.log 2>&1
echo "R1 rc=$? end $(date +%H:%M:%S) traces=$(ls output/full90e_nounit/traces/ 2>/dev/null | wc -l)"

echo "=== R2 [2] unit NL START $(date +%H:%M:%S) ==="
python3 run.py --max-steps 8 --units-path $UNITS --run-id full90e_unit_nl > output/full90e_unit_nl.log 2>&1
echo "R2 rc=$? end $(date +%H:%M:%S) traces=$(ls output/full90e_unit_nl/traces/ 2>/dev/null | wc -l)"

echo "=== R3 [3] unit KV START $(date +%H:%M:%S) ==="
python3 run.py --max-steps 8 --units-path $UNITS --kv-prefix-cache --run-id full90e_unit_kv > output/full90e_unit_kv.log 2>&1
echo "R3 rc=$? end $(date +%H:%M:%S) traces=$(ls output/full90e_unit_kv/traces/ 2>/dev/null | wc -l)"
echo "=== E-RUN ALL DONE $(date +%H:%M:%S) ==="

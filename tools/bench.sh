#!/bin/bash
# Benchmark matrix of the report. One job at a time; each configuration writes results/<name>/ (arrays + meta.json).
# Paper grid: the paper's algorithm and tolerances, timed REPS times (results/<name>_r<k>/meta.json keeps every run).
# Dense grids: 500 x 121 from the paper's starting point; 1000 x 121 and 2000 x 121 by continuation from the next
# coarser solution of the same back end and precision, with dt_max scaled to the debt step and a price tolerance of 1e-6.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venvs/torch/bin/python}
KM=./bin/km
COMMON="--params inputs/params.txt --quiet"
REPS=${REPS:-3}
run() {  # name reps args...
  local name=$1 reps=$2; shift 2
  for k in $(seq 1 $reps); do
    $KM $COMMON --out results/${name}_r$k "$@"
  done
  rm -rf results/$name; mv results/${name}_r$reps results/$name
  for k in $(seq 1 $((reps-1))); do mkdir -p results/$name/reps; mv results/${name}_r$k/meta.json results/$name/reps/meta_r$k.json; rm -rf results/${name}_r$k; done
  echo "$name done"
}
# ---- paper grid 201 x 75
run p_cpu64_exact $REPS --backend cpu --fp 64 --init inputs/init.bin --chunk_rows 0 --sweep_tol 1e-13
run p_cpu64 $REPS --backend cpu --fp 64 --init inputs/init.bin --sweep_tol 1e-9
run p_gpu64 $REPS --backend gpu --fp 64 --init inputs/init.bin --sweep_tol 1e-9
run p_cpu32 $REPS --backend cpu --fp 32 --init inputs/init.bin
run p_gpu32 $REPS --backend gpu --fp 32 --init inputs/init.bin
# ---- dense grids
D500="--nB 500 --nz 121"; D1000="--nB 1000 --nz 121 --dt_max 0.25 --tolq 1e-6"; D2000="--nB 2000 --nz 121 --dt_max 0.125 --tolq 1e-6"
run d500_cpu64_exact 1 --backend cpu --fp 64 --init inputs/init.bin $D500 --chunk_rows 0 --sweep_tol 1e-13
for bk in cpu gpu; do for fp in 64 32; do
  st=""; [ $fp = 64 ] && st="--sweep_tol 1e-9"
  run d500_${bk}${fp} 2 --backend $bk --fp $fp --init inputs/init.bin $D500 $st
  $PY tools/to_init.py results/d500_${bk}${fp} inputs/init_d500_${bk}${fp}.bin
  run d1000_${bk}${fp} 2 --backend $bk --fp $fp --init inputs/init_d500_${bk}${fp}.bin $D1000 $st
  $PY tools/to_init.py results/d1000_${bk}${fp} inputs/init_d1000_${bk}${fp}.bin
  run d2000_${bk}${fp} 2 --backend $bk --fp $fp --init inputs/init_d1000_${bk}${fp}.bin $D2000 $st
done; done
echo ALL DONE

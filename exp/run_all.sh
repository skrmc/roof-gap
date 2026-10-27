#!/bin/bash
# Runs every A100 experiment in sequence from this directory; one log per step in $RESULTS_DIR.
# Needs root for clock locking (set SM_CLOCK=none otherwise) and Nsight Systems for E3.
set -u
cd "$(dirname "$0")" || exit 1
export RESULTS_DIR="${RESULTS_DIR:-$PWD/../results}" MODEL_DIR="${MODEL_DIR:-$PWD/../models/llama31-8b}"
NSYS_DIR="$RESULTS_DIR/nsys"
mkdir -p "$RESULTS_DIR" "$NSYS_DIR"
step() { echo "=== $1 $(date +%T)"; }

step e1_gemm;     python e1_gemm.py > "$RESULTS_DIR/e1.log" 2>&1
step e3_nsys;     nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi \
                    --capture-range-end=repeat:8 -o "$NSYS_DIR/e3_%n" -f true python e3_nsys.py > "$RESULTS_DIR/e3.log" 2>&1
mapfile -t reps < <(printf '%s\n' "$NSYS_DIR"/e3_*.nsys-rep | sort -V)  # capture order: e3_1, e3_2, ...
for f in "${reps[@]}"; do nsys export --type sqlite -f true -o "${f%.nsys-rep}.sqlite" "$f" > /dev/null; done
step e3_analyze;  python e3_analyze.py "${reps[@]/%.nsys-rep/.sqlite}" > "$RESULTS_DIR/e3_analyze.log" 2>&1
step e2_grid;     python e2_steps.py > "$RESULTS_DIR/e2_grid.log" 2>&1
step e4_attn;     python e4_attn.py > "$RESULTS_DIR/e4.log" 2>&1 && python e4_attn.py --split >> "$RESULTS_DIR/e4.log" 2>&1
step finished

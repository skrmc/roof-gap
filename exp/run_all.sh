#!/bin/bash
# Runs every formal experiment in sequence on the Colab VM; one log per step, DONE marker at the end.
cd /content/exp
mkdir -p /content/results /content/nsys
step() { echo "=== $1 $(date +%T)" >> /content/results/progress.txt; }

step e1_gemm;      python e1_gemm.py > /content/results/e1.log 2>&1
step e3_nsys;      nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi \
                     --capture-range-end=repeat:8 -o /content/nsys/e3_%n -f true python e3_nsys.py > /content/results/e3.log 2>&1
step e3_export;    for f in $(ls /content/nsys/e3_*.nsys-rep | sort -V); do nsys export --type sqlite -f true -o "${f%.nsys-rep}.sqlite" "$f" > /dev/null 2>&1; done
step e3_analyze;   python e3_analyze.py $(ls /content/nsys/e3_*.sqlite | sort -V) > /content/results/e3_analyze.log 2>&1
step e2_grid;      python e2_steps.py graph 1215 all > /content/results/e2_grid.log 2>&1
step e4_eager;     python e2_steps.py eager 1215 sub > /content/results/e4_eager.log 2>&1
step e5_unlocked;  python e2_steps.py graph none sub > /content/results/e5_unlocked.log 2>&1
step DONE

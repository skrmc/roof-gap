"""E3: per-kernel decode breakdown. One cudaProfilerApi capture range per (b, s); run under
  nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=repeat:N
The CPU-side purity of every engine step is saved so the analysis can align GPU steps with it.
"""
import json

import torch

from common import dump, make_llm, run_steps, set_clock

CONFIGS = [(1, 512), (1, 4096), (8, 1024), (32, 1024), (32, 4096), (128, 512), (128, 1024), (256, 512)]

if __name__ == "__main__":
    set_clock(1215)
    llm = make_llm()
    run_steps(llm, 4, 128, osl=8, tag="warm")
    rows = []
    for b, s in CONFIGS:
        purity = []
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        r = run_steps(llm, b, s, tag="nsys", on_step=lambda i, p: purity.append(p))
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        r["purity"] = purity
        rows.append(r)
        print("ROW", json.dumps({k: v for k, v in r.items() if k not in ("pure_ms", "purity")}), flush=True)
    dump("/content/results/e3_cpu_steps.jsonl", rows)
    set_clock(None)

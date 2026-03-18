"""E2/E4/E5: per-step decode time over the (b, s) grid.

  python e2_steps.py graph 1215 all    # main grid: CUDA graphs, SM locked at 1215 MHz
  python e2_steps.py eager 1215 sub    # E4: no CUDA graphs, to size launch overhead
  python e2_steps.py graph none sub    # E5: unlocked clocks, sensitivity check
"""
import json
import sys

from common import GRID, ClockMonitor, dump, make_llm, run_steps, set_clock

mode, clock, points = sys.argv[1], sys.argv[2], sys.argv[3]
SUB = [(1, 512), (1, 4096), (8, 1024), (32, 1024), (32, 4096), (128, 512), (256, 512)]
grid = GRID if points == "all" else SUB
set_clock(None if clock == "none" else int(clock))
llm = make_llm(enforce_eager=(mode == "eager"))
run_steps(llm, 4, 128, osl=8, tag="warm")
rows = []
for b, s in grid:
    with ClockMonitor() as mon:
        r = run_steps(llm, b, s, tag=f"{mode}{clock}")
    r.update(mode=mode, clock=clock, clocks=mon.summary())
    rows.append(r)
    print("ROW", json.dumps({k: v for k, v in r.items() if k != "pure_ms"}), flush=True)
    dump(f"/content/results/e2_{mode}_{clock}_{points}.jsonl", rows)
set_clock(None)

"""E2: per-step decode time over the (b, s) grid, CUDA graphs on, SM clock from SM_CLOCK."""
import json

from common import GRID, SM_CLOCK, ClockMonitor, dump, make_llm, out_path, run_steps, set_clock

set_clock(SM_CLOCK)
llm = make_llm()
run_steps(llm, 4, 128, osl=8, tag="warm")
rows = []
for b, s in GRID:
    with ClockMonitor() as mon:
        r = run_steps(llm, b, s, tag="e2")
    r.update(clock=SM_CLOCK, clocks=mon.summary())
    rows.append(r)
    print("ROW", json.dumps({k: v for k, v in r.items() if k != "pure_ms"}), flush=True)
    dump(out_path("e2_steps.jsonl"), rows)
set_clock(None)

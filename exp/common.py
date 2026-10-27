"""Shared helpers for the A100 experiments.

Paths and clocks come from the environment:
  MODEL_DIR    local copy of Llama-3.1-8B (default ./models/llama31-8b)
  RESULTS_DIR  where raw results are written (default ./results)
  SM_CLOCK     MHz to lock the SM clock to, or "none" (default 1215: the highest clock that never hits
               the A100-40GB power cap). Locking needs root; without it the experiments run unlocked.
"""

import json
import os
import random
import statistics
import subprocess
import threading
import time

MODEL = os.environ.get("MODEL_DIR", "./models/llama31-8b")
RESULTS = os.environ.get("RESULTS_DIR", "./results")
SM_CLOCK = os.environ.get("SM_CLOCK", "1215")
OSL = 64
BS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SS = [128, 256, 512, 1024, 2048, 4096, 8192]
GRID = [(b, s) for b in BS for s in SS if b * (s + OSL) <= 150_000]  # same grid as analysis/baselines.py


def smi(args):
    return subprocess.run(f"nvidia-smi {args}", shell=True, capture_output=True, text=True).stdout.strip()


def set_clock(mhz):
    """Lock the SM clock to mhz, or reset it (None or "none")."""
    smi("-rgc" if mhz in (None, "none") else f"-lgc {mhz},{mhz}")


def out_path(name):
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


class ClockMonitor:
    """Samples SM clock, power and clock-event (throttle) reasons every 100 ms."""

    Q = "clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active"

    def __init__(self):
        self.samples, self._stop = [], threading.Event()

    def _run(self):
        p = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={self.Q}", "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE,
            text=True,
        )
        while not self._stop.is_set():
            line = p.stdout.readline()
            if line:
                sm, pw, t, reasons = [x.strip() for x in line.split(",")]
                self.samples.append((float(sm), float(pw), float(t), reasons))
        p.terminate()

    def __enter__(self):
        self.samples.clear()
        self._stop.clear()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.th.join(timeout=2)

    def summary(self):
        busy = [s for s in self.samples if s[1] > 100] or self.samples
        if not busy:
            return {}
        return {
            "sm_min": min(s[0] for s in busy),
            "sm_med": statistics.median(s[0] for s in busy),
            "power_max": max(s[1] for s in busy),
            "temp_max": max(s[2] for s in busy),
            "reasons": sorted({s[3] for s in busy}),
        }


def make_llm(max_model_len=8192 + 2 * OSL, **kw):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # drive the engine step by step in this process
    from vllm import LLM

    return LLM(
        MODEL,
        dtype="bfloat16",
        enable_prefix_caching=False,
        max_model_len=max_model_len,
        max_num_seqs=256,
        gpu_memory_utilization=0.9,
        seed=0,
        **kw,
    )


def run_steps(llm, b, s, osl=OSL + 1, tag="x", on_step=None):
    """Submit b requests of exactly s random prompt tokens, step the engine to completion, time every step.

    A step is pure decode at batch b when, before it runs, every request already has >= 1 output token and none
    has finished; prefill ramp-up and tail steps are excluded regardless of the chunked-prefill setting.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    eng = llm.llm_engine
    rng = random.Random(hash((b, s)) & 0xFFFF)
    sp = SamplingParams(max_tokens=osl, min_tokens=osl, ignore_eos=True, temperature=0.0)
    for i in range(b):
        eng.add_request(
            f"{tag}-{b}-{s}-{i}", TokensPrompt(prompt_token_ids=[rng.randint(1000, 120000) for _ in range(s)]), sp
        )
    ntok, steps, t_prev = {}, [], time.perf_counter()
    while eng.has_unfinished_requests():
        pure = len(ntok) == b and all(0 < v < osl for v in ntok.values())
        if on_step:
            on_step(len(steps), pure)
        outs = eng.step()
        t = time.perf_counter()
        for o in outs:
            ntok[o.request_id] = len(o.outputs[0].token_ids)
        steps.append((t - t_prev, pure))
        t_prev = t
    pure_ms = sorted(dt * 1e3 for dt, p in steps if p)
    n = len(pure_ms)
    return {
        "b": b,
        "s": s,
        "n_steps": len(steps),
        "n_pure": n,
        "step_ms": statistics.median(pure_ms) if n else float("nan"),
        "p10_ms": pure_ms[n // 10] if n else float("nan"),
        "p90_ms": pure_ms[9 * n // 10] if n else float("nan"),
        "pure_ms": [round(x, 4) for x in pure_ms],
    }


def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

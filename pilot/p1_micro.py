# Pilot step 1: attainable ceilings (HBM bandwidth, BF16 GEMM peak) and a decode-shaped GEMM sweep,
# with cold-L2 timing, under unlocked and locked SM clocks. Clocks/power/throttle reasons are sampled
# throughout so every number can be checked for throttling.
import json
import statistics
import subprocess
import sys
import threading
import time

import torch
import torch.nn.functional as F

DEV = torch.device("cuda")
SPEC = {"bw_GBps": 1555.0, "bf16_TFLOPs": 312.0}  # A100-SXM4-40GB datasheet
L2_FLUSH = torch.empty(256 * 2**20, dtype=torch.int8, device=DEV)  # 256 MiB > 40 MiB L2


def smi(args):
    return subprocess.run(f"nvidia-smi {args}", shell=True, capture_output=True, text=True).stdout.strip()


class ClockMonitor:
    """Samples SM clock, power, temperature and active clock-event (throttle) reasons every 50 ms."""

    Q = "clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active"

    def __init__(self):
        self.samples, self._stop = [], threading.Event()

    def _run(self):
        p = subprocess.Popen(["nvidia-smi", f"--query-gpu={self.Q}", "--format=csv,noheader,nounits", "-lms", "50"],
                             stdout=subprocess.PIPE, text=True)
        while not self._stop.is_set():
            line = p.stdout.readline()
            if line:
                sm, mem, pw, t, reasons = [x.strip() for x in line.split(",")]
                self.samples.append((float(sm), float(mem), float(pw), float(t), reasons))
        p.terminate()

    def __enter__(self):
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.th.join(timeout=2)

    def summary(self):
        if not self.samples:
            return {}
        sm = [s[0] for s in self.samples]
        busy = [s for s in self.samples if s[2] > 100] or self.samples  # samples while GPU is loaded
        return {"sm_min": min(sm), "sm_med_busy": statistics.median(s[0] for s in busy),
                "mem_clk": self.samples[-1][1], "power_max": max(s[2] for s in self.samples),
                "temp_max": max(s[3] for s in self.samples),
                "reasons_busy": sorted({s[4] for s in busy})}


def bench(fn, iters=50, warmup=10, flush=True):
    """Median kernel time in us; each iteration timed separately, L2 flushed outside the timed region."""
    for _ in range(warmup):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush:
            L2_FLUSH.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends)) * 1e3


def ceilings():
    n = 2 * 2**30 // 2  # 2 GiB of bf16
    a = torch.randn(n, device=DEV, dtype=torch.bfloat16)
    b = torch.empty_like(a)
    a32 = a.float()[: n // 2]  # 2 GiB fp32 for the read-only reduction
    out = {}
    t = bench(lambda: b.copy_(a), iters=20)
    out["copy_GBps"] = 2 * a.numel() * 2 / t / 1e3
    t = bench(lambda: a32.sum(), iters=20)
    out["read_GBps"] = a32.numel() * 4 / t / 1e3
    t = bench(lambda: b.fill_(1.0), iters=20)
    out["write_GBps"] = b.numel() * 2 / t / 1e3
    del a, b, a32
    for sz in (4096, 8192):
        x = torch.randn(sz, sz, device=DEV, dtype=torch.bfloat16)
        w = torch.randn(sz, sz, device=DEV, dtype=torch.bfloat16)
        t = bench(lambda: x @ w, iters=30)
        out[f"gemm{sz}_TFLOPs"] = 2 * sz**3 / t / 1e6
    return out


# Llama-3.1-8B linear layers as (name, N, K); vLLM fuses QKV and gate/up.
SHAPES = [("qkv", 6144, 4096), ("o", 4096, 4096), ("gate_up", 28672, 4096), ("down", 4096, 14336),
          ("lm_head", 128256, 4096)]
MS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 512]


def gemm_sweep(peak, flush=True, shapes=SHAPES, ms=MS):
    rows = []
    for name, N, K in shapes:
        w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)
        for M in ms:
            x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
            t_us = bench(lambda: F.linear(x, w), flush=flush)
            flops, nbytes = 2 * M * N * K, 2 * (N * K + M * K + M * N)
            roof = lambda bw, tf: max(flops / (tf * 1e6), nbytes / (bw * 1e3))  # us
            rows.append({"op": name, "M": M, "N": N, "K": K, "t_us": t_us, "flush": flush,
                         "TFLOPs": flops / t_us / 1e6, "GBps": nbytes / t_us / 1e3,
                         "eff_spec": roof(SPEC["bw_GBps"], SPEC["bf16_TFLOPs"]) / t_us,
                         "eff_attain": roof(peak["read_GBps"], peak["gemm8192_TFLOPs"]) / t_us})
    return rows


def run(tag):
    res = {"tag": tag}
    with ClockMonitor() as mon:
        res["ceil"] = ceilings()
    res["ceil_clocks"] = mon.summary()
    with ClockMonitor() as mon:
        res["sweep"] = gemm_sweep(res["ceil"])
    res["sweep_clocks"] = mon.summary()
    # L2 warm vs cold on the matrices that fit in the 40 MiB L2 (o: 32 MiB)
    res["sweep_warm"] = gemm_sweep(res["ceil"], flush=False, shapes=SHAPES[:2], ms=[1, 16, 64, 256])
    return res


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.get_device_name(), flush=True)
    results = []
    smi("-rgc")
    results.append(run("unlocked"))
    for clk in (1410, 1215):
        smi(f"-lgc {clk},{clk}")
        results.append(run(f"lock{clk}"))
    smi("-rgc")
    json.dump(results, open("/content/p1_micro.json", "w"), indent=1)
    for r in results:
        print(f"\n### {r['tag']}\nceilings: {json.dumps({k: round(v, 1) for k, v in r['ceil'].items()})}")
        print(f"clocks(ceil): {r['ceil_clocks']}\nclocks(sweep): {r['sweep_clocks']}")
        for row in r["sweep"]:
            if row["M"] in (1, 16, 64, 128, 256, 512):
                print(f"  {row['op']:8s} M={row['M']:4d} {row['t_us']:8.1f}us {row['GBps']:7.0f}GB/s "
                      f"{row['TFLOPs']:6.1f}TF eff_spec={row['eff_spec']:.2f} eff_attain={row['eff_attain']:.2f}")
        for row in r["sweep_warm"]:
            cold = next(c for c in r["sweep"] if c["op"] == row["op"] and c["M"] == row["M"])
            print(f"  warm-L2 {row['op']:4s} M={row['M']:3d} {row['t_us']:7.1f}us vs cold {cold['t_us']:7.1f}us")
    sys.stdout.flush()

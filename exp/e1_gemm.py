"""E1: decode-shaped GEMM efficiency, measured the way vLLM runs it.

Fixes from the pilot: weights rotate over enough copies to exceed L2 several times (clean, realistic cold L2:
the previous layer's weights occupy L2), and R back-to-back launches are captured in one CUDA graph so CPU
launch latency is not timed. Also re-measures the HBM/GEMM ceilings in the same torch/cuBLAS as vLLM.
"""
import json
import statistics

import torch
import torch.nn.functional as F

from common import ClockMonitor, set_clock

DEV = torch.device("cuda")
SHAPES = [("qkv", 6144, 4096), ("o", 4096, 4096), ("gate_up", 28672, 4096), ("down", 4096, 14336),
          ("lm_head", 128256, 4096)]
MS = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 320, 384, 512]


def graph_time_us(fn, reps):
    """Per-launch time of fn(i) for i in range(reps), captured as one CUDA graph; median of 15 replays."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(reps):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fn(i)
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(15):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps)
    return statistics.median(ts)


def ceilings():
    n = 2**30  # 2 GiB of bf16 / 4 GiB fp32 views
    a = torch.randn(n, device=DEV, dtype=torch.bfloat16)
    out = {}
    t = graph_time_us(lambda i: torch.sum(a, dtype=torch.float32), 3)
    out["read_GBps"] = n * 2 / t / 1e3
    b = torch.empty_like(a)
    t = graph_time_us(lambda i: b.copy_(a), 3)
    out["copy_GBps"] = 2 * n * 2 / t / 1e3
    del a, b
    x = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
    t = graph_time_us(lambda i: x @ w, 10)
    out["gemm8192_TFLOPs"] = 2 * 8192**3 / t / 1e6
    return out


def sweep():
    rows = []
    for name, N, K in SHAPES:
        wbytes = N * K * 2
        ncopy = max(3, -(-(160 * 2**20) // wbytes))  # >= 160 MiB of distinct weights, 4x the 40 MiB L2
        ws = [torch.randn(N, K, device=DEV, dtype=torch.bfloat16) for _ in range(ncopy)]
        reps = ncopy * max(1, 24 // ncopy)
        for M in MS:
            x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
            t = graph_time_us(lambda i: F.linear(x, ws[i % ncopy]), reps)
            nbytes = wbytes + 2 * M * (N + K)
            rows.append({"op": name, "M": M, "N": N, "K": K, "t_us": t, "GBps": nbytes / t / 1e3,
                         "TFLOPs": 2 * M * N * K / t / 1e6})
        del ws
        torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    set_clock(1215)
    with ClockMonitor() as mon:
        res = {"ceil": ceilings(), "sweep": sweep()}
    res["clocks"] = mon.summary()
    set_clock(None)
    json.dump(res, open("/content/results/e1_gemm.json", "w"), indent=1)
    print("CEIL", json.dumps({k: round(v, 1) for k, v in res["ceil"].items()}), res["clocks"])
    for r in res["sweep"]:
        if r["M"] in (1, 16, 64, 128, 256, 512):
            print(f"{r['op']:8s} M={r['M']:4d} {r['t_us']:8.1f}us {r['GBps']:6.0f}GB/s {r['TFLOPs']:6.1f}TF")

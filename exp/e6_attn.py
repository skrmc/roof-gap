"""E6: decode attention microbenchmark with vLLM's own kernel, called the way vLLM 0.32 calls it on A100.

FA2 paged split-KV (num_splits=0: on FA2 vLLM leaves the split count to the kernel heuristic), KV cache in vLLM's layout: one [blocks, H_kv, 16, 2*D]
tensor per layer, viewed as K and V via transpose(1, 2).split(D). KV copies rotate so every call reads a cold
KV (as in the model, where 31 other layers' KV sit between two uses of one layer's KV); timed in a CUDA graph.
"""
import json
import math
import statistics

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

from common import GRID, OSL, ClockMonitor, set_clock

DEV = torch.device("cuda")
HQ, HKV, D, BLK = 32, 8, 128, 16


def make_case(b, ctx, ncopy, max_k=None):
    nblk_seq = math.ceil(ctx / BLK)
    kvs = [torch.randn(b * nblk_seq, HKV, BLK, 2 * D, device=DEV, dtype=torch.bfloat16) for _ in range(ncopy)]
    views = [kv.transpose(1, 2).split(D, dim=-1) for kv in kvs]
    width = math.ceil((max_k or ctx) / BLK)  # vLLM's block table is max_model_len / block_size wide
    block_table = torch.zeros(b, max(width, nblk_seq), device=DEV, dtype=torch.int32)
    block_table[:, :nblk_seq] = torch.arange(b * nblk_seq, device=DEV, dtype=torch.int32).view(b, nblk_seq)
    q = torch.randn(b, HQ, D, device=DEV, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    cu_q = torch.arange(b + 1, device=DEV, dtype=torch.int32)
    seqused = torch.full((b,), ctx, device=DEV, dtype=torch.int32)

    def call(i):
        k, v = views[i % ncopy]
        flash_attn_varlen_func(q=q, k=k, v=v, out=out, cu_seqlens_q=cu_q, max_seqlen_q=1, seqused_k=seqused,
                               max_seqlen_k=max_k or ctx, softmax_scale=D**-0.5, causal=True, block_table=block_table,
                               fa_version=2, num_splits=0)  # 0 = kernel heuristic, as vLLM passes on FA2
    return call, kvs


def graph_time_us(fn, reps):
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
        a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(e) * 1e3 / reps)
    return statistics.median(ts)


def split_check():
    """Does the frozen max_seqlen_k explain the in-vLLM attention time? With full CUDA graphs vLLM captures the
    attention launch once, so FA2's host-side split heuristic may see max_model_len instead of the real context."""
    rows = []
    for b, s in [(1, 512), (1, 4096), (8, 1024), (32, 1024), (32, 4096), (128, 512), (128, 1024), (256, 512)]:
        ctx = s + OSL // 2
        for max_k in (ctx, 8192 + 2 * OSL, 131072):  # real, our max_model_len, Llama-3.1 default
            call, kvs = make_case(b, ctx, 3, max_k)
            rows.append({"b": b, "s": s, "max_seqlen_k": max_k, "t_us": graph_time_us(call, 24)})
            print("SPLIT", json.dumps(rows[-1]), flush=True)
            del kvs, call
    json.dump(rows, open("/content/results/e6_split.json", "w"), indent=1)


if __name__ == "__main__" and "split" in __import__("sys").argv:
    set_clock(1215)
    split_check()
    set_clock(None)
elif __name__ == "__main__":
    set_clock(1215)
    rows = []
    with ClockMonitor() as mon:
        for b, s in GRID:
            ctx = s + OSL // 2  # mean context over the measured decode steps
            kv_bytes = b * ctx * 2 * HKV * D * 2
            ncopy = max(3, min(16, math.ceil(160 * 2**20 / kv_bytes)))
            call, kvs = make_case(b, ctx, ncopy)
            t = graph_time_us(call, ncopy * max(1, 32 // ncopy))
            rows.append({"b": b, "s": s, "ctx": ctx, "t_us": t, "kv_MB": kv_bytes / 1e6, "GBps": kv_bytes / t / 1e3})
            print("ROW", json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in rows[-1].items()}), flush=True)
            del kvs, call
            torch.cuda.empty_cache()
    set_clock(None)
    json.dump({"rows": rows, "clocks": mon.summary()}, open("/content/results/e6_attn.json", "w"), indent=1)

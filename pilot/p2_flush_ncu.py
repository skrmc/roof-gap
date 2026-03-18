# Pilot step 2: (a) does the L2-flush method bias cold timings? compare a dirty flush (zero_ writes 256 MiB,
# leaving dirty lines that are written back during the timed kernel), a clean flush (read 256 MiB), and
# rotating weight copies (closest to a real model, where the previous layer's weights occupy L2).
# (b) can ncu read hardware counters on this VM? profile o_proj at M=1 for grid size and DRAM throughput.
import statistics
import subprocess

import torch
import torch.nn.functional as F

DEV = torch.device("cuda")
buf = torch.ones(256 * 2**20 // 4, dtype=torch.float32, device=DEV)


def timed(fn, pre, iters=50):
    for _ in range(10):
        pre(0)
        fn(0)
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        pre(i)
        s[i].record()
        fn(i)
        e[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in zip(s, e)) * 1e3


subprocess.run("nvidia-smi -lgc 1215,1215", shell=True, capture_output=True)
for name, N, K in [("qkv", 6144, 4096), ("o", 4096, 4096), ("gate_up", 28672, 4096)]:
    nrot = max(2, (256 * 2**20) // (N * K * 2) + 1)  # enough copies to exceed L2 several times
    ws = [torch.randn(N, K, device=DEV, dtype=torch.bfloat16) for _ in range(nrot)]
    for M in (1, 64):
        x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
        one = lambda i: F.linear(x, ws[0])
        rot = lambda i: F.linear(x, ws[i % nrot])
        res = {
            "warm": timed(one, lambda i: None),
            "dirty_flush": timed(one, lambda i: buf.zero_()),
            "clean_flush": timed(one, lambda i: buf.sum()),
            "rotating": timed(rot, lambda i: None),
        }
        print(f"{name:8s} M={M:3d} " + "  ".join(f"{k}={v:6.1f}us" for k, v in res.items()), flush=True)
    del ws

# (b) ncu on o_proj M=1
open("/content/ncu_target.py", "w").write(
    "import torch, torch.nn.functional as F\n"
    "w=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); x=torch.randn(1,4096,device='cuda',dtype=torch.bfloat16)\n"
    "for _ in range(3): F.linear(x,w)\ntorch.cuda.synchronize()\n")
r = subprocess.run(
    "ncu --target-processes all -k regex:'gemm|gemv|Kernel|cutlass|sm80' -c 2 "
    "--metrics launch__grid_size,launch__block_size,gpu__time_duration.sum,"
    "dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed "
    "python /content/ncu_target.py", shell=True, capture_output=True, text=True)
print("ncu rc", r.returncode)
print("\n".join(l for l in (r.stdout + r.stderr).splitlines() if not l.startswith("==PROF==") or "ERR" in l)[-3000:])
subprocess.run("nvidia-smi -rgc", shell=True, capture_output=True)

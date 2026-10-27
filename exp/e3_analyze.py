"""E3 analysis: split each nsys capture into GPU steps, keep pure decode steps, attribute kernel time to ops.

  python e3_analyze.py nsys/e3_1.sqlite ... (in capture order)  ->  $RESULTS_DIR/e3_breakdown.jsonl

Steps are delimited by the sampler kernel (one per forward pass). GPU steps are aligned with the CPU-side purity
list from e3_cpu_steps.jsonl when the counts agree; otherwise the steady-state middle of the run is used.
GEMMs are named by position: a pure step launches 4 per layer (qkv, o, gate_up, down) plus lm_head.
"""

import collections
import json
import os
import re
import sqlite3
import statistics
import sys

N_LAYERS = 32
GEMM = re.compile(r"gemm|cutlass|xmma|Kernel2", re.I)
MARKER = re.compile(r"^_gumbel_sample_kernel$")  # the sampler: exactly one per forward pass
EMBED = re.compile(r"embedding")  # first kernel of the forward pass; everything before it is step input prep
CATS = [
    ("attention", re.compile(r"flash|attn|paged", re.I)),
    ("kv_write", re.compile(r"reshape_and_cache", re.I)),
    ("norm", re.compile(r"rms_norm|layernorm", re.I)),
    ("act", re.compile(r"silu|gelu", re.I)),
]


def kernels(path):
    con = sqlite3.connect(path)
    q = (
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id ORDER BY k.start"
    )
    return con.execute(q).fetchall()


def split_steps(ks):
    steps, cur = [], []
    for k in ks:
        cur.append(k)
        if MARKER.search(k[2]):
            steps.append(cur)
            cur = []
    return steps


def union_ns(ks):
    busy, end = 0, -1
    for s, e, _ in ks:
        if e > end:
            busy += e - max(s, end)
            end = e
    return busy


def attribute(step):
    """Per-op time (ns) for one step; GEMMs by position, split-K reduce kernels go to the preceding GEMM."""
    gemm_idx = [i for i, k in enumerate(step) if GEMM.search(k[2]) and "splitK" not in k[2]]
    named = len(gemm_idx) == 4 * N_LAYERS + 1
    ops = ["qkv", "o", "gate_up", "down"]
    out, last_gemm = collections.Counter(), None
    after_lm_head = False
    first = next((i for i, k in enumerate(step) if EMBED.search(k[2])), 0)
    for i, (s, e, name) in enumerate(step):
        d = e - s
        if i in gemm_idx:
            j = gemm_idx.index(i)
            last_gemm = (ops[j % 4] if j < 4 * N_LAYERS else "lm_head") if named else "gemm"
            after_lm_head = named and j == 4 * N_LAYERS
            out[last_gemm] += d
        elif "splitK" in name and last_gemm:
            out[last_gemm] += d
        elif i < first:
            out["prep"] += d
        elif after_lm_head:
            out["sampling"] += d
        else:
            out[next((c for c, rx in CATS if rx.search(name)), "other")] += d
    return out, named


if __name__ == "__main__":
    res = os.environ.get("RESULTS_DIR", "./results")
    cpu = [json.loads(line) for line in open(f"{res}/e3_cpu_steps.jsonl")]
    rows = []
    for path, c in zip(sys.argv[1:], cpu, strict=True):
        steps = split_steps(kernels(path))
        purity = c["purity"]
        if abs(len(steps) - len(purity)) <= 1:
            pure_idx = [i for i in range(min(len(steps), len(purity))) if purity[i]]
            how = "aligned"
        else:  # steady-state middle: drop the prefill ramp at both ends
            ramp = len(steps) - (65 - 1)
            pure_idx = list(range(max(ramp, 0) + 2, len(steps) - max(ramp, 0) - 2))
            how = "middle"
        nk = collections.Counter(len(steps[i]) for i in pure_idx)
        mode_nk = nk.most_common(1)[0][0]
        pure_idx = [i for i in pure_idx if len(steps[i]) == mode_nk][1:]  # first pure step can include graph warmup
        per = []
        for i in pure_idx:
            st = steps[i]
            wall = st[-1][1] - steps[i - 1][-1][1]  # marker end to marker end; i >= 1 since we dropped the first
            ops, named = attribute(st)
            per.append({"wall": wall, "busy": union_ns(st), "ksum": sum(e - s for s, e, _ in st), **ops})
        keys = sorted({k for p in per for k in p})
        med = {k: statistics.median(p.get(k, 0) for p in per) / 1e6 for k in keys}  # ms
        names = collections.Counter()
        for i in pure_idx:
            for s, e, n in steps[i]:
                names[n] += e - s
        top = [(n[:90], round(t / len(pure_idx) / 1e3, 1)) for n, t in names.most_common(15)]
        row = {
            "b": c["b"],
            "s": c["s"],
            "how": how,
            "n_gpu_steps": len(steps),
            "n_cpu_steps": len(purity),
            "n_pure_used": len(pure_idx),
            "kernels_per_step": mode_nk,
            "gemms_named": named,
            "cpu_step_ms": c["step_ms"],
            **{f"{k}_ms": round(v, 4) for k, v in med.items()},
            "top_kernels_us": top,
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "top_kernels_us"}), flush=True)
    with open(f"{res}/e3_breakdown.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

"""Error-decomposition ladder: from the datasheet roofline to the measured decode step, one term at a time.

  R0  roofline, datasheet ceilings (312 TFLOP/s, 1555 GB/s)
  R1  roofline, attainable ceilings measured in E1 (HBM read, large GEMM)
  R2  R1 with every linear layer replaced by its measured microbenchmark time at M = b (E1)
  R3  R2 with attention replaced by the vLLM FA2 kernel microbenchmark (E4)
  measured wall time per step (E2); per-op kernel time inside vLLM (E3, nsys) validates R2/R3 op by op
Each gap R(k+1) - R(k) is one error source: ceilings, GEMM efficiency, attention; the rest is small kernels + idle.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from roofline import A100_SPEC, HW, LLAMA31_8B, decode_ops, step_time  # noqa: E402

RES = os.path.join(ROOT, "results")
OSL = 64
LINEAR = ("qkv", "o", "gate_up", "down", "lm_head")


def load_jsonl(name):
    return pd.DataFrame([json.loads(line) for line in open(os.path.join(RES, name))])


def gemm_lookup(e1):
    sw = pd.DataFrame(e1["sweep"])
    return {op: (g["M"].to_numpy(), g["t_us"].to_numpy()) for op, g in sw.groupby("op")}


def r2_time(b, s, attain, gl):
    """Linear layers from the microbenchmark (interpolated in M), everything else at the attainable roofline."""
    t = 0.0
    for name, f, B in decode_ops(LLAMA31_8B, b, s):
        if name in LINEAR:
            M, T = gl[name]
            per_call = np.interp(b, M, T) * 1e-6
            t += per_call * (1 if name == "lm_head" else LLAMA31_8B.L)
        else:
            t += max(f / attain.flops, B / attain.bw)
    return t


def attn_lookup():
    """Measured attention time per step (ms) from the E4 microbenchmark, keyed by (b, s); None if not run yet."""
    path = os.path.join(RES, "e4_attn.json")
    if not os.path.exists(path):
        return None
    return {(r["b"], r["s"]): r["t_us"] * LLAMA31_8B.L / 1e3 for r in json.load(open(path))["rows"]}


def fit_attn_model():
    """Least-squares fit of per-layer attention time t = t0 + KV_bytes / bw_eff on the E4 rows."""
    rows = json.load(open(os.path.join(RES, "e4_attn.json")))["rows"]
    x = np.array([r["kv_MB"] * 1e6 for r in rows])
    y = np.array([r["t_us"] * 1e-6 for r in rows])
    slope, t0 = np.polyfit(x, y, 1)
    return t0, 1 / slope


def grid_table():
    e1 = json.load(open(os.path.join(RES, "e1_gemm.json")))
    attain = HW("attainable", e1["ceil"]["gemm8192_TFLOPs"] * 1e12, e1["ceil"]["read_GBps"] * 1e9)
    gl = gemm_lookup(e1)
    al = attn_lookup()
    e2 = load_jsonl("e2_steps.jsonl")
    rows = []
    for r in e2.itertuples():
        s_eff = r.s + OSL // 2
        row = {
            "b": r.b,
            "s": r.s,
            "measured": r.step_ms,
            "p10": r.p10_ms,
            "p90": r.p90_ms,
            "n_pure": r.n_pure,
            "R0_spec": step_time(LLAMA31_8B, A100_SPEC, r.b, s_eff) * 1e3,
            "R1_attain": step_time(LLAMA31_8B, attain, r.b, s_eff) * 1e3,
            "R2_gemm": r2_time(r.b, s_eff, attain, gl) * 1e3,
        }
        if al:  # R3: attention from the microbenchmark instead of the roofline
            attn_roof = (
                next(
                    max(f / attain.flops, B / attain.bw)
                    for n, f, B in decode_ops(LLAMA31_8B, r.b, s_eff)
                    if n == "attn"
                )
                * 1e3
            )
            row["R3_attn"] = row["R2_gemm"] - attn_roof + al[(r.b, r.s)]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["ratio_R0"] = df.measured / df.R0_spec
    return df, attain


def breakdown_table(attain):
    """Per-op measured kernel time (E3) next to the per-op roofline at the attainable ceilings."""
    e3 = load_jsonl("e3_breakdown.jsonl")
    rows = []
    for r in e3.to_dict("records"):
        ops = {
            n: max(f / attain.flops, B / attain.bw) * 1e3
            for n, f, B in decode_ops(LLAMA31_8B, r["b"], r["s"] + OSL // 2)
        }
        meas_gemm = sum(r.get(f"{k}_ms", 0) for k in LINEAR + ("gemm",))
        small = sum(r.get(f"{k}_ms", 0) for k in ("kv_write", "norm", "act", "other", "sampling", "prep"))
        rows.append(
            {
                "b": r["b"],
                "s": r["s"],
                "wall": r.get("wall_ms"),
                "busy": r.get("busy_ms"),
                "gemm_meas": meas_gemm,
                "gemm_roof": sum(ops[k] for k in LINEAR),
                "attn_meas": r.get("attention_ms", 0),
                "attn_roof": ops["attn"],
                "small_meas": small,
                "small_roof": ops["elementwise"] + ops["kv_write"],
                "idle": (r.get("wall_ms") or 0) - (r.get("busy_ms") or 0),
                "cpu_step_ms": r["cpu_step_ms"],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    g, attain = grid_table()
    print(f"attainable: {attain.bw / 1e9:.0f} GB/s, {attain.flops / 1e12:.0f} TFLOP/s")
    print(g.round(3).to_string(index=False))
    g.to_csv(os.path.join(RES, "ladder_grid.csv"), index=False)
    bd = breakdown_table(attain)
    print(bd.round(3).to_string(index=False))
    bd.to_csv(os.path.join(RES, "ladder_breakdown.csv"), index=False)

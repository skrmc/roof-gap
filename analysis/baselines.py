"""Decode step-time predictions of every baseline on one (b, s) grid -> results/baselines.csv.

A step at prompt length s with OSL generated tokens is predicted at the mean context s + OSL/2, which matches
how the measurement averages over pure decode steps (and aiconfigurator's TPOT over the OSL).

Needs GenZ, aisimulate and a checkout of LLM-Viewer (LLMVIEWER_DIR, default third_party/LLM-Viewer); see README.
"""

import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from roofline import A100_SPEC, LLAMA31_8B, step_time  # noqa: E402

warnings.filterwarnings("ignore")
OSL = 64
BS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SS = [128, 256, 512, 1024, 2048, 4096, 8192]
GRID = [(b, s) for b in BS for s in SS if b * (s + OSL) <= 150_000]
HF_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"


def hand(b, s):
    return step_time(LLAMA31_8B, A100_SPEC, b, s + OSL // 2) * 1e3


def genz_fn():
    from GenZ import decode_moddeling

    def run(b, s, system):
        return decode_moddeling(
            model="meta-llama/meta-llama-3.1-8b",
            batch_size=b,
            input_tokens=s,
            output_tokens=OSL,
            Bb=1,
            system_name=system,
            bits="bf16",
            model_profilling=False,
        )["Latency"]

    return lambda b, s: run(b, s, "A100_40GB_GPU")


def llmviewer_fn():
    cwd = os.getcwd()
    os.chdir(os.environ.get("LLMVIEWER_DIR", os.path.join(ROOT, "third_party", "LLM-Viewer")))
    sys.path.insert(0, os.getcwd())
    from model_analyzer import ModelAnalyzer

    analyzer = ModelAnalyzer(HF_ID, "nvidia_A100_40G")
    os.chdir(cwd)

    def run(b, s):
        r = analyzer.analyze(seqlen=s + OSL // 2, batchsize=b, w_bit=16, a_bit=16, kv_bit=16, use_flashattention=True)
        return r["total_results"]["decode"]["inference_time"] * 1e3

    return run


def aic_systems_dir():
    """aiconfigurator's a100_sxm is the 80GB part; give it our A100-40GB spec next to its bundled A100 data."""
    import aiconfigurator_core

    d = tempfile.mkdtemp(prefix="aic_systems_")
    shutil.copy(os.path.join(ROOT, "baselines", "a100_sxm_40g.yaml"), d)
    os.makedirs(os.path.join(d, "data"))
    os.symlink(
        os.path.join(os.path.dirname(aiconfigurator_core.__file__), "systems", "data", "a100_sxm"),
        os.path.join(d, "data", "a100_sxm"),
    )
    return d


AIC_BIN = shutil.which("aiconfigurator") or os.path.join(os.path.dirname(sys.executable), "aiconfigurator")


def aic(b, s, mode, systems):
    cmd = [
        AIC_BIN,
        "cli",
        "estimate",
        "--systems-paths",
        systems,
        "--model-path",
        HF_ID,
        "--system",
        "a100_sxm_40g",
        "--backend",
        "vllm",
        "--estimate-mode",
        "static_gen",
        "--batch-size",
        str(b),
        "--isl",
        str(s),
        "--osl",
        str(OSL),
        "--tp-size",
        "1",
        "--database-mode",
        mode,
        "--log-level",
        "ERROR",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r"TPOT:\s+([\d.]+) ms", out)
    return float(m.group(1)) if m else float("nan")


if __name__ == "__main__":
    genz = genz_fn()
    lv = llmviewer_fn()
    rows = [
        {
            "b": b,
            "s": s,
            "hand_roofline": hand(b, s),
            "llmviewer_fa": lv(b, s),
            "genz": genz(b, s),
        }
        for b, s in GRID
    ]
    systems = aic_systems_dir()
    with ThreadPoolExecutor(8) as ex:
        for mode in ("SOL", "EMPIRICAL", "SILICON"):
            for row, t in zip(rows, ex.map(lambda bs, mode=mode: aic(*bs, mode, systems), GRID), strict=True):
                row[f"aic_{mode.lower()}"] = t
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", "baselines.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print(f"wrote {len(rows)} rows to {path}")

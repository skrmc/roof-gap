# Pilot step 3: vLLM decode step time via the differential method, against the hand-derived roofline.
#   T_decode_step = (T(out=N) - T(out=1)) / (N - 1)
# All b requests are submitted together with distinct random prompts of exactly s tokens, so after the
# prefill steps every scheduler step is a pure decode batch of b. Prefix caching is off.
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, "/content")
from roofline import A100_SPEC, HW, LLAMA31_8B, step_time  # noqa: E402

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

MODEL = "/content/llama31-8b"
chunked = {"default": None, "off": False}[sys.argv[1] if len(sys.argv) > 1 else "default"]
kw = {} if chunked is None else {"enable_chunked_prefill": chunked}
llm = LLM(MODEL, dtype="bfloat16", enable_prefix_caching=False, max_model_len=8192, max_num_seqs=256,
          gpu_memory_utilization=0.9, seed=0, **kw)
cfg = llm.llm_engine.vllm_config
print("CONFIG", json.dumps({
    "enable_chunked_prefill": cfg.scheduler_config.enable_chunked_prefill,
    "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
    "prefix_caching": cfg.cache_config.enable_prefix_caching,
    "cudagraph_mode": str(cfg.compilation_config.cudagraph_mode),
    "cudagraph_sizes_head": list(cfg.compilation_config.cudagraph_capture_sizes or [])[:8],
    "max_cudagraph": cfg.compilation_config.max_cudagraph_capture_size,
    "kv_cache_blocks": getattr(cfg.cache_config, "num_gpu_blocks", None),
}, default=str), flush=True)


def gen(b, s, out, seed):
    rng = random.Random(seed)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randint(1000, 120000) for _ in range(s)]) for _ in range(b)]
    sp = SamplingParams(max_tokens=out, min_tokens=out, ignore_eos=True, temperature=0.0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()
    assert all(len(r.outputs[0].token_ids) == out for r in res)
    return time.perf_counter() - t0


ATTAIN = HW("A100 attainable (pilot)", 271e12, 1431e9)
N = 65
gen(4, 128, 8, 0)  # warmup
rows = []
for b, s in [(1, 512), (1, 4096), (32, 1024), (128, 512)]:
    t1, tn = [], []
    for rep in range(3):
        t1.append(gen(b, s, 1, rep))
        tn.append(gen(b, s, N, rep))
    step = (statistics.median(tn) - statistics.median(t1)) / (N - 1)
    s_eff = s + N // 2
    pred_spec, pred_att = step_time(LLAMA31_8B, A100_SPEC, b, s_eff), step_time(LLAMA31_8B, ATTAIN, b, s_eff)
    row = {"b": b, "s": s, "step_ms": step * 1e3, "pred_spec_ms": pred_spec * 1e3, "pred_attain_ms": pred_att * 1e3,
           "meas/spec": step / pred_spec, "meas/attain": step / pred_att,
           "t1_spread": (max(t1) - min(t1)) / statistics.median(t1), "tn_spread": (max(tn) - min(tn)) / statistics.median(tn)}
    rows.append(row)
    print("ROW", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
json.dump(rows, open(f"/content/p3_vllm_{sys.argv[1] if len(sys.argv) > 1 else 'default'}.json", "w"), indent=1)

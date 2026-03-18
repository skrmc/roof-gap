# Pilot step 5: per-step timing by driving the engine manually. A step is "pure decode at batch b" when every
# request already has >= 1 output token and none has finished, so prefill ramp-up and tail steps are excluded
# regardless of the chunked-prefill setting.
import json
import os
import random
import statistics
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

llm = LLM("/content/llama31-8b", dtype="bfloat16", enable_prefix_caching=False, max_model_len=8192,
          max_num_seqs=256, gpu_memory_utilization=0.9, seed=0)
eng = llm.llm_engine


def run(b, s, out, tag):
    rng = random.Random(0)
    sp = SamplingParams(max_tokens=out, min_tokens=out, ignore_eos=True, temperature=0.0)
    for i in range(b):
        eng.add_request(f"{tag}-{i}", TokensPrompt(prompt_token_ids=[rng.randint(1000, 120000) for _ in range(s)]), sp)
    ntok, steps, t_prev = {}, [], time.perf_counter()
    while eng.has_unfinished_requests():
        before = dict(ntok)
        outs = eng.step()
        t = time.perf_counter()
        for o in outs:
            ntok[o.request_id] = len(o.outputs[0].token_ids)
        pure = len(before) == b and all(0 < v < out for v in before.values())
        steps.append((t - t_prev, pure))
        t_prev = t
    pure = [dt for dt, p in steps if p]
    return {"b": b, "s": s, "steps": len(steps), "pure_steps": len(pure),
            "pure_med_ms": statistics.median(pure) * 1e3, "pure_p10_ms": sorted(pure)[len(pure) // 10] * 1e3,
            "pure_p90_ms": sorted(pure)[9 * len(pure) // 10] * 1e3}


run(4, 128, 8, "warm")
for b, s in [(1, 512), (32, 1024), (128, 512)]:
    print("STEP", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in run(b, s, 65, f"{b}x{s}").items()}),
          flush=True)

# Pilot step 4: per-kernel decode breakdown. Two cudaProfilerApi capture ranges (out=1, out=N) in one
# process; per-kernel decode time = (sum_N - sum_1) / (N - 1). Run under
#   nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=repeat:2
import os
import random
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # keep the model runner in this process
import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

b, s, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
llm = LLM("/content/llama31-8b", dtype="bfloat16", enable_prefix_caching=False, max_model_len=8192,
          max_num_seqs=256, gpu_memory_utilization=0.9, seed=0)


def gen(out):
    rng = random.Random(0)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randint(1000, 120000) for _ in range(s)]) for _ in range(b)]
    llm.generate(prompts, SamplingParams(max_tokens=out, min_tokens=out, ignore_eos=True, temperature=0.0),
                 use_tqdm=False)


gen(8)  # warmup
for out in (1, N):
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    t0 = time.perf_counter()
    gen(out)
    torch.cuda.synchronize()
    print(f"WALL out={out} {time.perf_counter() - t0:.4f}s", flush=True)
    torch.cuda.profiler.stop()

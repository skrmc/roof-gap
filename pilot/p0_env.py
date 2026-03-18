# Pilot step 0: probe hardware, software and permissions on the Colab VM.
import json
import os
import subprocess


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.returncode, (r.stdout + r.stderr).strip())


checks = {
    "whoami": "whoami; id",
    "gpu": "nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,"
           "power.limit,power.max_limit,clocks.max.sm,clocks.max.mem,clocks.sm,clocks.mem,"
           "ecc.mode.current,persistence_mode,mig.mode.current --format=csv",
    "app_clocks": "nvidia-smi -q -d CLOCK | sed -n '1,60p'",
    "supported_clocks_head": "nvidia-smi -q -d SUPPORTED_CLOCKS | sed -n '1,25p'",
    "lock_sm_clock": "nvidia-smi -lgc 1410,1410",
    "lock_mem_clock": "nvidia-smi -lmc 1215,1215",
    "reset_sm_clock": "nvidia-smi -rgc",
    "reset_mem_clock": "nvidia-smi -rmc",
    "persistence": "nvidia-smi -pm 1",
    "cuda": "nvcc --version | tail -2; ls -d /usr/local/cuda*",
    "nsys": "which nsys; nsys --version",
    "ncu": "which ncu; ncu --version; ls /usr/local/cuda/bin | grep -i -E 'ncu|nsys'",
    "perf_counter_perm": "cat /proc/driver/nvidia/params | grep -i RmProfilingAdminOnly",
    "cpu": "lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|NUMA'; free -g | head -2",
    "disk": "df -h /content | tail -1",
    "python": "python -V; pip list 2>/dev/null | grep -i -E '^(torch|triton|transformers|vllm|flashinfer|numpy) '",
}
out = {k: sh(v) for k, v in checks.items()}
for k, (rc, txt) in out.items():
    print(f"===== {k} (rc={rc})\n{txt}\n")
json.dump(out, open("/content/p0_env.json", "w"), indent=1)

# Install vLLM into an isolated venv in the background so the system torch used by microbenchmarks is untouched.
import subprocess
cmd = ("which uv || pip install -q uv; "
       "uv venv -q --python 3.12 /content/venv-vllm && "
       "VIRTUAL_ENV=/content/venv-vllm uv pip install vllm 'huggingface_hub[hf_transfer]' pandas && "
       "/content/venv-vllm/bin/python -c 'import vllm, torch; print(\"VLLM_OK\", vllm.__version__, torch.__version__, torch.version.cuda)'")
subprocess.Popen(f"( {cmd} ) > /content/install_vllm.log 2>&1", shell=True, start_new_session=True)
print("started")

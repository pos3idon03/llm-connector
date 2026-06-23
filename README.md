# llm-connector

A Poetry-managed template for running [vLLM](https://github.com/vllm-project/vllm) natively — no Docker, no GPU-passthrough overhead — with an OpenAI-compatible HTTP API.

## Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.11 |
| Poetry | 1.8+ |
| NVIDIA drivers + CUDA | CUDA 12.0 |
| **or** AMD ROCm | 6.2 |

Install Poetry if you don't have it:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

---

## Install

### Linux / WSL2

The bootstrap script auto-detects your GPU and runs the correct install:

```bash
make install
# or: bash setup.sh
```

What it does:
1. Detects `nvidia-smi` (NVIDIA) or `rocminfo`/`rocm-smi` (AMD)
2. Verifies driver version requirements
3. **NVIDIA:** `poetry install --with nvidia` — vLLM CUDA wheels resolve cleanly via PyPI
4. **AMD:** `poetry install` (base deps) → `pip install torch` from the ROCm 6.2 index → `pip install vllm` — Poetry bypassed for GPU packages because its resolver can't handle PyTorch ROCm's transitive dependencies

### Windows

Requires WSL2. Run in PowerShell:

```powershell
.\setup.ps1
```

This converts the path and delegates to `setup.sh` inside your WSL2 distro. GPU drivers for WSL2:
- NVIDIA: [CUDA on WSL](https://developer.nvidia.com/cuda/wsl)
- AMD: [ROCm on Windows](https://rocm.docs.amd.com/en/latest/deploy/windows)

---

## Configure

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```bash
MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct

# Required for gated models (Llama, Mistral, etc.)
HF_TOKEN=hf_...
```

All available options are documented in [`.env.example`](.env.example).

---

## Run

```bash
make run
# or: poetry run llm-connector
```

The server starts on `http://0.0.0.0:8000` by default (override with `HOST`/`PORT` in `.env`).

---

## Verify

```bash
make test
# runs curl against /health and /v1/models with pretty-printed output
# override port: make test PORT=9000
```

Or manually:

```bash
curl http://localhost:8000/health       # {"status":"ok","engine_ready":true}
curl http://localhost:8000/v1/models
```

`engine_ready` is `false` while the model is still loading — wait and retry.

---

## Settings UI

Open **`http://localhost:8000/settings`** in a browser while the server is running.

The form reads your current `.env` values and writes changes back to the file on save — no rebuild needed. A restart is still required to reload the engine with new settings (e.g. a different `MODEL_ID` or `GPU_MEMORY_UTILIZATION`).

To change config without a browser:

```bash
cp .env.example .env   # first time
nano .env              # edit directly
```

---

## Model caching

vLLM downloads models from HuggingFace Hub on first run and caches them. Override the cache location with `HF_HOME`:

```bash
# In .env or your shell profile
HF_HOME=/data/hf_cache
```

The tokenizer and model weights share the same cache path — no double download.

---

## Using with an OpenAI client

Point any OpenAI-compatible client at `http://localhost:8000/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Streaming works too:

```python
stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## Known limitations

- No authentication. Add an API key middleware before exposing beyond localhost.
- Single model only. To serve multiple models, run separate instances on different ports.
- Client disconnect does not abort in-flight generation (`engine.abort()` not wired).
- AMD ROCm wheel availability depends on the vLLM release; update the `--extra-index-url` version in `setup.sh` when upgrading vLLM.

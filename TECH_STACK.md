# Tech Stack — llm-connector

## Overview

llm-connector is a native bare-metal LLM server. No Docker. It runs a local language model via vLLM and exposes an OpenAI-compatible HTTP API with an agentic tool-calling loop backed by MCP servers.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 6 GB | 8 GB+ |
| System RAM | 16 GB | 32 GB |
| Disk (model cache) | 10 GB | 50 GB+ |
| CPU | Any x86-64 | Modern multi-core |

**GPU support:**
- **NVIDIA** — CUDA 12.0+. Tested on RTX 5060 Ti (Blackwell, SM 12.x) with `ENFORCE_EAGER=true`.
- **AMD** — ROCm 6.2+. Installed via pip post-`poetry install` (PyPI does not carry ROCm wheels).

**Notes:**
- Blackwell / SM 12.x GPUs require `ENFORCE_EAGER=true` until vLLM adds native SM 12.x support (requires CUDA ≥ 12.9 for compute capability detection).
- VRAM budget: model weights + KV cache. A 3B model (~6 GB) on an 8 GB card requires `MAX_MODEL_LEN ≤ 16384` to leave enough KV cache headroom.

---

## Runtime Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11 – 3.12 |
| Inference engine | vLLM | 0.23.0 |
| Deep learning | PyTorch | 2.11.0+cu130 |
| Attention kernel | FlashAttention | 2 |
| Web framework | FastAPI | 0.138.0 |
| ASGI server | Uvicorn | 0.49.0 |
| Config management | pydantic-settings | 2.14.2 |
| Data validation | Pydantic | 2.13.4 |
| Env file handling | python-dotenv | 1.2.2 |
| MCP client SDK | mcp | 1.28.0 |
| Package manager | Poetry | 1.8+ |

---

## Architecture

```
Browser
  │
  ├─ GET  /           → Chat UI (inline HTML/JS, SSE stream consumer)
  ├─ GET  /settings   → Settings UI (reads/writes .env)
  ├─ GET  /mcp        → MCP server manager UI
  │
  ├─ POST /v1/chat/completions   → Direct vLLM passthrough (no tools)
  └─ POST /v1/chat/agentic       → Tool-calling loop (vLLM + MCP)
          │
          ├─ apply_chat_template → raw prompt
          ├─ AsyncLLMEngine.generate()
          ├─ parse tool calls from output
          └─ MCPManager.call_tool() → MCP server → result → next turn
```

### Key components

**`src/llm_connector/config.py`** — pydantic-settings v2. All runtime config comes from `.env`. Fields: `MODEL_ID`, `GPU_MEMORY_UTILIZATION`, `TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, `DTYPE`, `QUANTIZATION`, `TRUST_REMOTE_CODE`, `ENFORCE_EAGER`, `HOST`, `PORT`.

**`src/llm_connector/main.py`** — FastAPI app. Lifespan initialises the vLLM engine and `MCPManager`. Serves the chat UI, settings UI, MCP manager UI, and all API endpoints.

**`src/llm_connector/mcp_client.py`** — Async MCP client. Each configured server runs in its own `asyncio.Task` keeping a persistent session. Supports both stdio (local commands) and Streamable HTTP transports.

**`src/llm_connector/schemas.py`** — OpenAI-compatible Pydantic models for request/response.

---

## MCP Integration

- Transport: **Streamable HTTP** (MCP spec 2025-03-26, POST-based) and **stdio** (local subprocess)
- Tool injection: tools from all connected MCP servers are merged into the chat template at inference time
- Tool-calling loop: up to 10 rounds per request; stops when the model produces no tool calls
- Tools toggle: `tool_choice: "none"` skips tool injection entirely (UI toggle: 🔧 button)

---

## Inference Pipeline

1. `apply_chat_template` — converts messages + tools into a raw string prompt
2. `SamplingParams` — temperature, top-p, max tokens
3. `AsyncLLMEngine.generate()` — vLLM V1 engine, async streaming
4. Delta streaming — each chunk yields `text[len(previous):]` as SSE
5. Tool call parsing — regex-based, supports Qwen `<tool_call>`, Llama `<|python_tag|>`, and bare JSON formats

---

## Bootstrap

| Script | Purpose |
|---|---|
| `setup.sh` | Linux/WSL2 — GPU detect → driver check → `poetry install --with nvidia` or AMD pip path |
| `setup.ps1` | Windows — thin WSL2 launcher |
| `Makefile` | `make install` / `make run` / `make test` |

---

## Known Constraints

- **Python 3.11–3.12 only** — triton has no wheels for 3.13+
- **Single model** — one vLLM engine per process; run multiple instances for multiple models
- **No authentication** — add API key middleware before exposing beyond localhost
- **Blackwell GPU** — `ENFORCE_EAGER=true` required; disables CUDA graphs and FlashInfer sampling
- **AMD ROCm** — Poetry cannot resolve `pytorch-triton-rocm` transitive deps; AMD install bypasses Poetry for GPU packages
- **Client disconnect** — does not abort in-flight generation (`engine.abort()` not wired)

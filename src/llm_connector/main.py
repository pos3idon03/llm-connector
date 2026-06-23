import html
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from dotenv import dotenv_values, set_key
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

_ENV_PATH = Path(".env")

from .config import settings
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    DeltaMessage,
    ModelCard,
    ModelList,
    UsageInfo,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine_args = AsyncEngineArgs(
        model=settings.model_id,
        gpu_memory_utilization=settings.gpu_memory_utilization,
        tensor_parallel_size=settings.tensor_parallel_size,
        max_model_len=settings.max_model_len,
        dtype=settings.dtype,
        quantization=settings.quantization,
        trust_remote_code=settings.trust_remote_code,
        enforce_eager=settings.enforce_eager,
    )
    if settings.enforce_eager:
        # FlashInfer checks CUDA arch in a subprocess — must be set before spawning
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    # ponytail: from_engine_args is synchronous and blocks during model load — expected
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine.start_background_loop()
    app.state.tokenizer = await engine.get_tokenizer()
    app.state.engine = engine
    app.state.engine_ready = True
    yield


app = FastAPI(title="llm-connector", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "engine_ready": getattr(app.state, "engine_ready", False)}


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelCard(id=settings.model_id)])


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # ponytail: no auth — add API key middleware if exposing beyond localhost
    engine: AsyncLLMEngine = app.state.engine
    tokenizer = app.state.tokenizer

    conversation = [{"role": m.role, "content": m.content} for m in request.messages]
    try:
        prompt: str = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"chat template error: {e}")

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 512,  # ponytail: default 512 — tune per model
        stop=request.stop if request.stop else None,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    result_generator: AsyncIterator = engine.generate(prompt, sampling_params, request_id)

    if request.stream:
        return StreamingResponse(
            _stream_response(result_generator, request_id, created, request.model),
            media_type="text/event-stream",
        )

    final_output = None
    async for output in result_generator:
        final_output = output

    if final_output is None:
        raise HTTPException(status_code=500, detail="engine returned no output")

    out = final_output.outputs[0]
    return ChatCompletionResponse(
        id=request_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=out.text),
                finish_reason=out.finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=len(final_output.prompt_token_ids),
            completion_tokens=len(out.token_ids),
            total_tokens=len(final_output.prompt_token_ids) + len(out.token_ids),
        ),
    )


async def _stream_response(
    result_generator: AsyncIterator,
    request_id: str,
    created: int,
    model: str,
) -> AsyncIterator[str]:
    # ponytail: engine.abort() on client disconnect not wired — add for production
    first_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role="assistant"))],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    previous_text = ""
    async for output in result_generator:
        text = output.outputs[0].text
        delta_text = text[len(previous_text):]
        previous_text = text
        if not delta_text:
            continue
        chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=DeltaMessage(content=delta_text),
                    finish_reason=output.outputs[0].finish_reason,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/settings", response_class=HTMLResponse)
async def settings_page():
    return _render_settings_page()


@app.post("/settings", response_class=HTMLResponse)
async def save_settings(request: Request):
    form = await request.form()
    if not _ENV_PATH.exists():
        _ENV_PATH.touch()
    for key in [
        "MODEL_ID", "GPU_MEMORY_UTILIZATION", "TENSOR_PARALLEL_SIZE",
        "MAX_MODEL_LEN", "DTYPE", "QUANTIZATION", "HOST", "PORT",
        "HF_TOKEN", "HF_HOME",
    ]:
        value = form.get(key, "")
        if value:
            set_key(str(_ENV_PATH), key, str(value))
        elif key in ("MAX_MODEL_LEN", "QUANTIZATION", "HF_TOKEN", "HF_HOME"):
            set_key(str(_ENV_PATH), key, "")  # allow clearing optional fields
    trust = "true" if form.get("TRUST_REMOTE_CODE") else "false"
    set_key(str(_ENV_PATH), "TRUST_REMOTE_CODE", trust)
    eager = "true" if form.get("ENFORCE_EAGER") else "false"
    set_key(str(_ENV_PATH), "ENFORCE_EAGER", eager)
    return _render_settings_page(saved=True)


def _render_settings_page(saved: bool = False) -> str:
    v = {
        "MODEL_ID": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "TENSOR_PARALLEL_SIZE": "1",
        "MAX_MODEL_LEN": "",
        "DTYPE": "auto",
        "QUANTIZATION": "",
        "TRUST_REMOTE_CODE": "false",
        "ENFORCE_EAGER": "false",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "HF_TOKEN": "",
        "HF_HOME": "",
    }
    if _ENV_PATH.exists():
        v.update(dotenv_values(_ENV_PATH))

    def e(k: str) -> str:
        return html.escape(v.get(k, ""))

    def opt(tag: str, label: str) -> str:
        sel = ' selected' if v.get("DTYPE") == tag else ''
        return f'<option value="{tag}"{sel}>{label}</option>'

    def qopt(tag: str) -> str:
        sel = ' selected' if v.get("QUANTIZATION") == tag else ''
        return f'<option value="{tag}"{sel}>{tag or "none"}</option>'

    notice = (
        '<p class="notice">Saved. <strong>Restart the server</strong> to apply changes.</p>'
        if saved else ""
    )
    trust_checked = 'checked' if v.get("TRUST_REMOTE_CODE", "").lower() == "true" else ""
    eager_checked = 'checked' if v.get("ENFORCE_EAGER", "").lower() == "true" else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>llm-connector · Settings</title>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:system-ui,sans-serif;background:#f4f4f5;margin:0;padding:2rem 1rem}}
  .card{{background:#fff;max-width:540px;margin:0 auto;padding:2rem;border-radius:10px;box-shadow:0 1px 6px rgba(0,0,0,.1)}}
  h1{{font-size:1.15rem;margin:0 0 .25rem}}
  .sub{{font-size:.82rem;color:#777;margin:0 0 1.5rem}}
  label{{display:block;font-size:.82rem;font-weight:600;color:#444;margin-bottom:.3rem}}
  .opt{{font-weight:400;color:#aaa}}
  input[type=text],input[type=number],input[type=password],select{{
    width:100%;padding:.45rem .6rem;border:1px solid #ddd;border-radius:5px;
    font-size:.88rem;margin-bottom:1rem;background:#fafafa}}
  input:focus,select:focus{{outline:2px solid #6366f1;border-color:transparent;background:#fff}}
  .check-row{{display:flex;align-items:center;gap:.5rem;margin-bottom:1rem}}
  .check-row input{{width:auto;margin:0}}
  .check-row label{{margin:0;font-weight:400}}
  .section{{font-size:.7rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
    color:#aaa;border-top:1px solid #eee;padding-top:1.2rem;margin:1.5rem 0 .9rem}}
  button{{background:#6366f1;color:#fff;border:none;padding:.55rem 1.4rem;
    border-radius:5px;font-size:.88rem;cursor:pointer;font-weight:600}}
  button:hover{{background:#4f46e5}}
  .notice{{background:#f0fdf4;border:1px solid #86efac;border-radius:5px;
    padding:.7rem 1rem;font-size:.84rem;color:#166534;margin-top:1.2rem}}
</style>
</head>
<body>
<div class="card">
  <h1>llm-connector</h1>
  <p class="sub">Changes are written to <code>.env</code>. Restart the server to apply.</p>
  <form method="post" action="/settings">

    <div class="section">Model</div>
    <label>MODEL_ID</label>
    <input type="text" name="MODEL_ID" value="{e('MODEL_ID')}">
    <label>HF_TOKEN <span class="opt">(gated models)</span></label>
    <input type="password" name="HF_TOKEN" value="{e('HF_TOKEN')}" placeholder="hf_...">
    <label>HF_HOME <span class="opt">(cache dir)</span></label>
    <input type="text" name="HF_HOME" value="{e('HF_HOME')}" placeholder="/data/hf_cache">

    <div class="section">Memory &amp; Parallelism</div>
    <label>GPU_MEMORY_UTILIZATION</label>
    <input type="number" name="GPU_MEMORY_UTILIZATION" step="0.01" min="0.1" max="1.0" value="{e('GPU_MEMORY_UTILIZATION')}">
    <label>TENSOR_PARALLEL_SIZE</label>
    <input type="number" name="TENSOR_PARALLEL_SIZE" min="1" value="{e('TENSOR_PARALLEL_SIZE')}">
    <label>MAX_MODEL_LEN <span class="opt">(leave blank for model default)</span></label>
    <input type="number" name="MAX_MODEL_LEN" min="1" value="{e('MAX_MODEL_LEN')}" placeholder="model default">

    <div class="section">Inference</div>
    <label>DTYPE</label>
    <select name="DTYPE">
      {opt('auto','auto')}{opt('float16','float16')}{opt('bfloat16','bfloat16')}{opt('float32','float32')}
    </select>
    <label>QUANTIZATION</label>
    <select name="QUANTIZATION">
      {qopt('')}{qopt('awq')}{qopt('gptq')}{qopt('squeezellm')}
    </select>
    <div class="check-row">
      <input type="checkbox" name="TRUST_REMOTE_CODE" value="true" {trust_checked}>
      <label>TRUST_REMOTE_CODE</label>
    </div>
    <div class="check-row">
      <input type="checkbox" name="ENFORCE_EAGER" value="true" {eager_checked}>
      <label>ENFORCE_EAGER <span class="opt">(disable CUDA graphs + FlashInfer — for new/unsupported GPUs)</span></label>
    </div>

    <div class="section">Server</div>
    <label>HOST</label>
    <input type="text" name="HOST" value="{e('HOST')}">
    <label>PORT</label>
    <input type="number" name="PORT" min="1" max="65535" value="{e('PORT')}">

    <button type="submit">Save to .env</button>
    {notice}
  </form>
</div>
</body>
</html>"""


def main():
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    assert settings.host  # self-check: config loads correctly
    main()

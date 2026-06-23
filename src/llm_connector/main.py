import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

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
        disable_log_requests=True,
    )
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


def main():
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    assert settings.host  # self-check: config loads correctly
    main()

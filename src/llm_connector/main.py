import html as _html
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger("uvicorn.error")

import uvicorn
from dotenv import dotenv_values, set_key
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.exceptions import VLLMValidationError

_ENV_PATH = Path(".env")

from .config import settings
from .mcp_client import MCPManager
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
    ToolCall,
    ToolCallFunction,
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
    app.state.tokenizer = engine.get_tokenizer()
    app.state.engine = engine
    app.state.engine_ready = True

    mcp = MCPManager()
    await mcp.load()
    app.state.mcp = mcp

    yield

    await mcp.shutdown()


app = FastAPI(title="llm-connector", lifespan=lifespan)


# ── Health / Models ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "engine_ready": getattr(app.state, "engine_ready", False)}


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelCard(id=settings.model_id)])


def _inject_datetime(messages: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc).strftime("%A, %d %B %Y %H:%M UTC")
    note = f"Today is {now}."
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": f"{note}\n\n{messages[0]['content']}"}] + messages[1:]
    return [{"role": "system", "content": note}] + messages


def _msg_to_dict(m: ChatMessage) -> dict:
    """Serialize a ChatMessage to the dict form apply_chat_template expects."""
    d: dict = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    return d


def _template_tools(tools: list[dict] | None) -> list[dict] | None:
    """Extract function defs from OpenAI-format tools for apply_chat_template."""
    if not tools:
        return None
    return [t["function"] for t in tools if t.get("type") == "function"] or None


# ── Chat completions (passthrough) ─────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # ponytail: no auth — add API key middleware if exposing beyond localhost
    logger.info("chat/completions input: %s", request.model_dump_json())
    engine: AsyncLLMEngine = app.state.engine
    tokenizer = app.state.tokenizer

    conversation = _inject_datetime([_msg_to_dict(m) for m in request.messages])
    tmpl_tools = _template_tools(request.tools)

    template_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if tmpl_tools:
        template_kwargs["tools"] = tmpl_tools
    try:
        prompt: str = tokenizer.apply_chat_template(conversation, **template_kwargs)
    except Exception:
        if tmpl_tools:
            template_kwargs.pop("tools")
            try:
                prompt = tokenizer.apply_chat_template(conversation, **template_kwargs)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"chat template error: {e}")
        else:
            raise

    output_reserve = request.max_tokens or 1024
    vllm_prompt: str | dict = prompt
    if settings.max_model_len:
        prompt_ids = tokenizer.encode(prompt)
        max_input = settings.max_model_len - output_reserve
        if len(prompt_ids) > max_input:
            logger.warning("prompt truncated: %d → %d tokens", len(prompt_ids), max_input)
            # ponytail: keep tail (recent turns), not head — system prompt is re-injected each request
            vllm_prompt = {"prompt_token_ids": prompt_ids[-max_input:]}

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=output_reserve,
        stop=request.stop if request.stop else None,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    result_generator: AsyncIterator = engine.generate(vllm_prompt, sampling_params, request_id)

    if request.stream:
        return StreamingResponse(
            _stream_response(result_generator, request_id, created, request.model),
            media_type="text/event-stream",
        )

    final_output = None
    try:
        async for output in result_generator:
            final_output = output
    except VLLMValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if final_output is None:
        raise HTTPException(status_code=500, detail="engine returned no output")

    out = final_output.outputs[0]
    logger.info("chat/completions output: %s", out.text)

    usage = UsageInfo(
        prompt_tokens=len(final_output.prompt_token_ids),
        completion_tokens=len(out.token_ids),
        total_tokens=len(final_output.prompt_token_ids) + len(out.token_ids),
    )

    calls = _parse_tool_calls(out.text) if tmpl_tools else []
    if calls:
        tool_calls = [
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                function=ToolCallFunction(
                    name=c["name"],
                    arguments=json.dumps(c["arguments"]) if isinstance(c["arguments"], dict) else str(c["arguments"]),
                ),
            )
            for c in calls
        ]
        response_msg = ChatMessage(role="assistant", content=None, tool_calls=tool_calls)
        finish_reason = "tool_calls"
    else:
        response_msg = ChatMessage(role="assistant", content=out.text)
        finish_reason = out.finish_reason

    return ChatCompletionResponse(
        id=request_id,
        created=created,
        model=request.model,
        choices=[ChatCompletionResponseChoice(index=0, message=response_msg, finish_reason=finish_reason)],
        usage=usage,
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

    logger.info("chat/completions stream output: %s", previous_text)
    yield "data: [DONE]\n\n"


# ── Agentic chat (MCP tool-calling loop) ──────────────────────────────────────

@app.post("/v1/chat/agentic")
async def agentic_chat(request: ChatCompletionRequest):
    logger.info("chat/agentic input: %s", request.model_dump_json())
    return StreamingResponse(
        _agentic_stream(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _build_prompt(tokenizer, messages: list[dict], tools: list[dict] | None) -> str:
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        if tools:
            kwargs.pop("tools")
            return tokenizer.apply_chat_template(messages, **kwargs)
        raise


def _parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from common LLM output formats."""
    calls: list[dict] = []

    # Qwen-style <tool_call>...</tool_call>
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        try:
            d = json.loads(m.group(1))
            if "name" in d:
                calls.append({"name": d["name"], "arguments": d.get("arguments", d.get("parameters", {}))})
        except json.JSONDecodeError:
            pass
    if calls:
        return calls

    # Llama 3.1 <|python_tag|>{...}
    for m in re.finditer(r"<\|python_tag\|>\s*(\{[^<]*\})", text):
        try:
            d = json.loads(m.group(1))
            if "name" in d:
                calls.append({"name": d["name"], "arguments": d.get("parameters", d.get("arguments", {}))})
        except json.JSONDecodeError:
            pass
    if calls:
        return calls

    # Bare JSON with name + arguments/parameters keys
    for m in re.finditer(r'\{"name"\s*:[^}]*(?:"arguments"|"parameters")[^}]*\}', text, re.DOTALL):
        try:
            d = json.loads(m.group(0))
            if "name" in d:
                calls.append({"name": d["name"], "arguments": d.get("arguments", d.get("parameters", {}))})
        except json.JSONDecodeError:
            pass
    return calls


async def _agentic_stream(request: ChatCompletionRequest) -> AsyncIterator[str]:
    engine: AsyncLLMEngine = app.state.engine
    tokenizer = app.state.tokenizer
    mcp: MCPManager = app.state.mcp

    messages = _inject_datetime([_msg_to_dict(m) for m in request.messages])
    tools = mcp.get_all_tools() if request.tool_choice != "none" else []
    if tools:
        messages[0]["content"] += (
            "\n\nOnly call tools when the request genuinely requires external real-time data "
            "(e.g. live stock prices, weather, external APIs). "
            "For math, reasoning, writing, or anything answerable from your training data, "
            "answer directly — do not use tools."
        )

    try:
        for _turn in range(10):  # ponytail: max 10 tool-call rounds, add config when needed
            prompt = _build_prompt(tokenizer, messages, tools or None)
            max_tokens = request.max_tokens or 512
            vllm_prompt: str | dict = prompt
            if settings.max_model_len:
                prompt_ids = tokenizer.encode(prompt)
                max_input = settings.max_model_len - max_tokens
                if len(prompt_ids) > max_input:
                    logger.warning("agentic prompt truncated: %d → %d tokens", len(prompt_ids), max_input)
                    vllm_prompt = {"prompt_token_ids": prompt_ids[-max_input:]}
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=max_tokens,
            )
            request_id = f"agentic-{uuid.uuid4().hex}"

            full_text = ""
            async for output in engine.generate(vllm_prompt, sampling_params, request_id):
                text = output.outputs[0].text
                delta = text[len(full_text):]
                full_text = text
                if delta:
                    yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

            if not tools:
                break

            calls = _parse_tool_calls(full_text)
            if not calls:
                break

            messages.append({"role": "assistant", "content": full_text})

            for call in calls:
                yield f"data: {json.dumps({'type': 'tool_call', 'name': call['name'], 'args': call['arguments']})}\n\n"
                try:
                    result = await mcp.call_tool(call["name"], call["arguments"])
                    yield f"data: {json.dumps({'type': 'tool_result', 'name': call['name'], 'content': result})}\n\n"
                    messages.append({"role": "tool", "content": result, "name": call["name"]})
                except Exception as e:
                    msg = str(e)
                    yield f"data: {json.dumps({'type': 'tool_result', 'name': call['name'], 'content': msg, 'error': True})}\n\n"
                    messages.append({"role": "tool", "content": f"Error: {msg}", "name": call["name"]})

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    logger.info("chat/agentic output: %s", full_text)
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ── Chat UI ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def chat_page():
    return _CHAT_HTML


_CHAT_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>llm-connector · Chat</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{display:flex;height:100vh;font-family:system-ui,sans-serif;overflow:hidden}
#sidebar{width:230px;background:#1e1e2e;color:#cdd6f4;padding:1rem;display:flex;flex-direction:column;gap:.7rem;overflow-y:auto;flex-shrink:0}
#sidebar h1{font-size:.95rem;font-weight:700;color:#fff;letter-spacing:.02em}
.nav-links{display:flex;gap:.8rem}
.nav-links a{color:#89b4fa;font-size:.8rem;text-decoration:none}
.nav-links a:hover{text-decoration:underline}
.sb-label{font-size:.72rem;color:#a6adc8;font-weight:600;text-transform:uppercase;letter-spacing:.07em}
#model-sel{width:100%;padding:.35rem .5rem;border-radius:5px;border:none;background:#313244;color:#cdd6f4;font-size:.82rem}
.divider{border:none;border-top:1px solid #313244}
#mcp-list{display:flex;flex-direction:column;gap:.35rem}
.mcp-item{font-size:.8rem;display:flex;align-items:flex-start;gap:.4rem}
.mcp-dot{font-size:.85rem;flex-shrink:0;margin-top:.05rem}
.mcp-dot.on{color:#a6e3a1}.mcp-dot.off{color:#f38ba8}
.mcp-info{display:flex;flex-direction:column;gap:.1rem}
.mcp-name{font-weight:600}
.mcp-tools{color:#a6adc8;font-size:.75rem}
.no-mcp{font-size:.8rem;color:#6c7086}
#new-chat{background:#313244;color:#cdd6f4;border:none;padding:.45rem;border-radius:5px;cursor:pointer;font-size:.82rem;margin-top:auto}
#new-chat:hover{background:#45475a}
#main{flex:1;display:flex;flex-direction:column;min-width:0;background:#f4f4f5}
#messages{flex:1;overflow-y:auto;padding:1.5rem;display:flex;flex-direction:column;gap:1rem}
.msg{display:flex}
.msg.user{justify-content:flex-end}
.msg.assistant{justify-content:flex-start}
.bubble{max-width:72%;padding:.7rem 1rem;border-radius:12px;font-size:.9rem;line-height:1.55;word-break:break-word}
.msg.user .bubble{background:#6366f1;color:#fff;border-bottom-right-radius:3px}
.msg.assistant .bubble{background:#fff;color:#1e1e2e;border-bottom-left-radius:3px;box-shadow:0 1px 4px rgba(0,0,0,.1)}
.text{white-space:pre-wrap}
.tool-card{margin-top:.5rem;border:1px solid #e2e8f0;border-radius:7px;overflow:hidden;font-size:.8rem}
.tool-card summary{padding:.4rem .65rem;cursor:pointer;background:#f8fafc;font-family:monospace;list-style:none;display:flex;align-items:baseline;gap:.4rem}
.tool-card summary::-webkit-details-marker{display:none}
.tc-icon{font-size:.85rem}
.tc-name{font-weight:600;color:#374151}
.tc-args{color:#9ca3af;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:260px}
.tc-result{padding:.5rem .65rem;font-family:monospace;color:#374151;white-space:pre-wrap;max-height:180px;overflow-y:auto;background:#fafafa;border-top:1px solid #e2e8f0}
.tc-result.err{color:#dc2626}
#input-wrap{padding:.85rem 1rem;border-top:1px solid #e2e8f0;background:#fff;display:flex;gap:.6rem;align-items:flex-end}
#inp{flex:1;padding:.6rem .75rem;border:1px solid #ddd;border-radius:8px;font-size:.9rem;resize:none;font-family:inherit;min-height:44px;max-height:140px;overflow-y:auto;line-height:1.5}
#inp:focus{outline:2px solid #6366f1;border-color:transparent}
#send-btn{background:#6366f1;color:#fff;border:none;padding:.6rem 1.3rem;border-radius:8px;font-size:.9rem;cursor:pointer;font-weight:600;height:44px;flex-shrink:0}
#send-btn:hover{background:#4f46e5}
#send-btn:disabled{background:#a5b4fc;cursor:not-allowed}
#tools-btn{background:#313244;color:#cdd6f4;border:none;padding:.6rem .75rem;border-radius:8px;font-size:.9rem;cursor:pointer;height:44px;flex-shrink:0;opacity:.45}
#tools-btn.on{background:#6366f1;color:#fff;opacity:1}
</style>
</head>
<body>
<div id="sidebar">
  <h1>llm-connector</h1>
  <div class="nav-links">
    <a href="/settings">Settings</a>
    <a href="/mcp">MCP Servers</a>
  </div>
  <hr class="divider">
  <span class="sb-label">Model</span>
  <select id="model-sel"></select>
  <hr class="divider">
  <span class="sb-label">MCP Servers</span>
  <div id="mcp-list"><span class="no-mcp">Loading…</span></div>
  <button id="new-chat" onclick="newChat()">+ New Chat</button>
</div>
<div id="main">
  <div id="messages"></div>
  <div id="input-wrap">
    <textarea id="inp" rows="1" placeholder="Ask something… (Enter to send, Shift+Enter for newline)"></textarea>
    <button id="tools-btn" onclick="toggleTools()" title="Toggle tool use">🔧</button>
    <button id="send-btn" onclick="sendMsg()">Send</button>
  </div>
</div>
<script>
let conversation = [];
let busy = false;
let useTools = false;
function toggleTools(){
  useTools=!useTools;
  const btn=document.getElementById('tools-btn');
  btn.classList.toggle('on',useTools);
  btn.title=useTools?'Tools ON — click to disable':'Tools OFF — click to enable';
}

fetch('/v1/models').then(r=>r.json()).then(d=>{
  const s=document.getElementById('model-sel');
  d.data.forEach(m=>{const o=document.createElement('option');o.value=m.id;o.text=m.id;s.appendChild(o);});
});

function refreshMCP(){
  fetch('/mcp/api/servers').then(r=>r.json()).then(servers=>{
    const div=document.getElementById('mcp-list');
    if(!servers.length){div.innerHTML='<span class="no-mcp">No servers configured</span>';return;}
    div.innerHTML=servers.map(s=>`
      <div class="mcp-item">
        <span class="mcp-dot ${s.connected?'on':'off'}">${s.connected?'●':'○'}</span>
        <div class="mcp-info">
          <span class="mcp-name">${esc(s.name)}</span>
          <span class="mcp-tools">${s.connected?s.tool_count+' tool'+(s.tool_count===1?'':'s'):'disconnected'}</span>
        </div>
      </div>`).join('');
  }).catch(()=>{});
}
refreshMCP();
setInterval(refreshMCP,5000);

function newChat(){conversation=[];document.getElementById('messages').innerHTML='';}

document.getElementById('inp').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg();}
});

async function sendMsg(){
  if(busy)return;
  const inp=document.getElementById('inp');
  const text=inp.value.trim();
  if(!text)return;

  inp.value='';
  busy=true;
  document.getElementById('send-btn').disabled=true;

  conversation.push({role:'user',content:text});
  appendMsg('user',text);

  const model=document.getElementById('model-sel').value;
  const {div:msgDiv,textEl}=appendAssistant();
  let assistantText='';

  try{
    const resp=await fetch('/v1/chat/agentic',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model,messages:conversation,temperature:0.7,tool_choice:useTools?'auto':'none'}),
    });
    const reader=resp.body.getReader();
    const dec=new TextDecoder();
    let buf='';
    while(true){
      const{done,value}=await reader.read();
      if(done)break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\\n');
      buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        let ev;
        try{ev=JSON.parse(line.slice(6));}catch{continue;}
        if(ev.type==='token'){
          assistantText+=ev.content;
          textEl.textContent=assistantText;
          msgDiv.scrollIntoView({block:'end'});
        }else if(ev.type==='tool_call'){
          addToolCard(msgDiv,ev);
        }else if(ev.type==='tool_result'){
          setToolResult(msgDiv,ev);
        }else if(ev.type==='error'){
          textEl.textContent+=(assistantText?'\\n':'')+'⚠️ '+ev.message;
        }else if(ev.type==='done'){
          if(assistantText)conversation.push({role:'assistant',content:assistantText});
        }
      }
    }
  }catch(e){
    textEl.textContent='[Connection error: '+e.message+']';
  }

  busy=false;
  document.getElementById('send-btn').disabled=false;
  inp.focus();
}

function appendMsg(role,content){
  const div=document.createElement('div');
  div.className='msg '+role;
  div.innerHTML='<div class="bubble"><span class="text">'+esc(content)+'</span></div>';
  document.getElementById('messages').appendChild(div);
  div.scrollIntoView({block:'end'});
}

function appendAssistant(){
  const div=document.createElement('div');
  div.className='msg assistant';
  const bubble=document.createElement('div');
  bubble.className='bubble';
  const textEl=document.createElement('span');
  textEl.className='text';
  bubble.appendChild(textEl);
  div.appendChild(bubble);
  document.getElementById('messages').appendChild(div);
  return{div,textEl};
}

function addToolCard(msgDiv,ev){
  const bubble=msgDiv.querySelector('.bubble');
  const det=document.createElement('details');
  det.className='tool-card';
  det.dataset.toolName=ev.name;
  const argsStr=JSON.stringify(ev.args);
  det.innerHTML=`<summary><span class="tc-icon">&#x1f527;</span><span class="tc-name">${esc(ev.name)}</span><span class="tc-args">${esc(argsStr)}</span></summary><div class="tc-result">Running…</div>`;
  det.open=true;
  bubble.appendChild(det);
  det.scrollIntoView({block:'end'});
}

function setToolResult(msgDiv,ev){
  const cards=[...msgDiv.querySelectorAll('.tool-card[data-tool-name]')];
  const card=cards.reverse().find(d=>d.dataset.toolName===ev.name);
  if(!card)return;
  const res=card.querySelector('.tc-result');
  res.textContent=ev.content;
  if(ev.error)res.classList.add('err');
  if(!ev.error)card.open=false;
}

function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
</script>
</body>
</html>"""


# ── MCP Config UI ──────────────────────────────────────────────────────────────

@app.get("/mcp", response_class=HTMLResponse)
async def mcp_page():
    return _MCP_HTML


_MCP_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>llm-connector · MCP Servers</title>
<style>
*{box-sizing:border-box}
body{font-family:system-ui,sans-serif;background:#f4f4f5;margin:0;padding:2rem 1rem}
.card{background:#fff;max-width:660px;margin:0 auto;padding:2rem;border-radius:10px;box-shadow:0 1px 6px rgba(0,0,0,.1)}
h1{font-size:1.15rem;margin:0 0 .2rem}
.sub{font-size:.82rem;color:#777;margin:0 0 1.5rem}
.sub a{color:#6366f1;text-decoration:none}
.section{font-size:.7rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#aaa;border-top:1px solid #eee;padding-top:1.2rem;margin:1.5rem 0 .9rem}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th{text-align:left;padding:.4rem .5rem;font-size:.75rem;color:#888;border-bottom:1px solid #eee}
td{padding:.5rem .5rem;border-bottom:1px solid #f0f0f0;vertical-align:top}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:3px;font-size:.75rem;font-weight:600}
.badge.ok{background:#dcfce7;color:#166534}
.badge.err{background:#fee2e2;color:#991b1b}
.badge.wait{background:#fef9c3;color:#854d0e}
.del-btn{background:none;border:1px solid #fca5a5;color:#dc2626;border-radius:4px;padding:.2rem .55rem;cursor:pointer;font-size:.8rem}
.del-btn:hover{background:#fef2f2}
.tools-list{font-size:.76rem;color:#555;margin-top:.25rem}
label{display:block;font-size:.82rem;font-weight:600;color:#444;margin-bottom:.3rem}
input[type=text],textarea{width:100%;padding:.45rem .6rem;border:1px solid #ddd;border-radius:5px;font-size:.88rem;margin-bottom:1rem;background:#fafafa;font-family:inherit}
input:focus,textarea:focus{outline:2px solid #6366f1;border-color:transparent;background:#fff}
textarea{resize:vertical}
.radio-row{display:flex;gap:1.5rem;margin-bottom:1rem}
.radio-row label{font-weight:400;display:flex;align-items:center;gap:.4rem;margin:0;cursor:pointer}
button[type=submit]{background:#6366f1;color:#fff;border:none;padding:.55rem 1.4rem;border-radius:5px;font-size:.88rem;cursor:pointer;font-weight:600}
button[type=submit]:hover{background:#4f46e5}
#notice{background:#f0fdf4;border:1px solid #86efac;border-radius:5px;padding:.7rem 1rem;font-size:.84rem;color:#166534;margin-top:1rem;display:none}
</style>
</head>
<body>
<div class="card">
  <h1>MCP Servers</h1>
  <p class="sub"><a href="/">&#x2190; Chat</a> &nbsp;&middot;&nbsp; <a href="/settings">Settings</a></p>

  <div class="section">Configured Servers</div>
  <div id="server-table"><em style="font-size:.85rem;color:#888">Loading…</em></div>

  <div class="section">Add Server</div>
  <form id="add-form">
    <label>Name</label>
    <input type="text" id="f-name" placeholder="my-server" required>

    <label>Transport</label>
    <div class="radio-row">
      <label><input type="radio" name="transport" value="stdio" checked onchange="toggleTransport()"> stdio &mdash; local command</label>
      <label><input type="radio" name="transport" value="http" onchange="toggleTransport()"> HTTP / SSE</label>
    </div>

    <div id="stdio-fields">
      <label>Command</label>
      <input type="text" id="f-command" placeholder="npx">
      <label>Args <span style="font-weight:400;color:#aaa">(one per line)</span></label>
      <textarea id="f-args" rows="3" placeholder="-y&#10;@modelcontextprotocol/server-filesystem&#10;/home/user"></textarea>
    </div>

    <div id="http-fields" style="display:none">
      <label>URL</label>
      <input type="text" id="f-url" placeholder="http://localhost:3000/sse">
    </div>

    <button type="submit">Add Server</button>
  </form>
  <div id="notice">Server added. Connecting in background…</div>
</div>
<script>
function toggleTransport(){
  const isStdio=document.querySelector('[name=transport]:checked').value==='stdio';
  document.getElementById('stdio-fields').style.display=isStdio?'':'none';
  document.getElementById('http-fields').style.display=isStdio?'none':'';
}

async function loadServers(){
  const resp=await fetch('/mcp/api/servers');
  const servers=await resp.json();
  const div=document.getElementById('server-table');
  if(!servers.length){div.innerHTML='<p style="font-size:.85rem;color:#888">No servers configured yet.</p>';return;}
  div.innerHTML=`<table>
    <tr><th>Name</th><th>Transport</th><th>Endpoint</th><th>Status</th><th>Tools</th><th></th></tr>
    ${servers.map(s=>`<tr>
      <td><strong>${esc(s.name)}</strong></td>
      <td>${esc(s.transport)}</td>
      <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:monospace;font-size:.78rem">${esc(s.endpoint)}</td>
      <td><span class="badge ${s.connected?'ok':'err'}">${s.connected?'connected':'disconnected'}</span></td>
      <td>
        ${s.tool_count}
        ${s.tools.length?`<div class="tools-list">${s.tools.map(t=>`<div title="${esc(t.description)}">&#x2022; ${esc(t.name)}</div>`).join('')}</div>`:''}
      </td>
      <td><button class="del-btn" onclick="delServer('${esc(s.name)}')">Delete</button></td>
    </tr>`).join('')}
  </table>`;
}

async function delServer(name){
  if(!confirm('Delete server "'+name+'"?'))return;
  await fetch('/mcp/api/servers/'+encodeURIComponent(name),{method:'DELETE'});
  loadServers();
}

document.getElementById('add-form').addEventListener('submit',async e=>{
  e.preventDefault();
  const transport=document.querySelector('[name=transport]:checked').value;
  const cfg={name:document.getElementById('f-name').value.trim(),transport};
  if(transport==='stdio'){
    cfg.command=document.getElementById('f-command').value.trim();
    cfg.args=document.getElementById('f-args').value.split('\\n').map(l=>l.trim()).filter(Boolean);
  }else{
    cfg.url=document.getElementById('f-url').value.trim();
  }
  await fetch('/mcp/api/servers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)});
  document.getElementById('add-form').reset();
  toggleTransport();
  const n=document.getElementById('notice');
  n.style.display='block';
  setTimeout(()=>n.style.display='none',4000);
  loadServers();
});

function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

loadServers();
setInterval(loadServers,5000);
</script>
</body>
</html>"""


# ── MCP JSON API ───────────────────────────────────────────────────────────────

@app.get("/mcp/api/servers")
async def mcp_list_servers():
    return app.state.mcp.servers_status()


@app.post("/mcp/api/servers", status_code=201)
async def mcp_add_server(cfg: dict):
    if not cfg.get("name"):
        raise HTTPException(status_code=422, detail="name is required")
    if cfg.get("transport") not in ("stdio", "http"):
        raise HTTPException(status_code=422, detail="transport must be 'stdio' or 'http'")
    await app.state.mcp.add_server(cfg)
    return {"ok": True}


@app.delete("/mcp/api/servers/{name}", status_code=200)
async def mcp_remove_server(name: str):
    await app.state.mcp.remove_server(name)
    return {"ok": True}


# ── Settings UI ────────────────────────────────────────────────────────────────

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
        return _html.escape(v.get(k, ""))

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
  .sub a{{color:#6366f1;text-decoration:none}}
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
  <p class="sub"><a href="/">&#x2190; Chat</a> &nbsp;&middot;&nbsp; Changes are written to <code>.env</code>. Restart the server to apply.</p>
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

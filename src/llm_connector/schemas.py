from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = 1.0
    max_tokens: int | None = None
    top_p: float = 1.0
    stop: str | list[str] | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "vllm"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]

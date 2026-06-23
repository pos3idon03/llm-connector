from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", env_ignore_empty=True, extra="ignore")

    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    max_model_len: int | None = None
    dtype: str = "auto"
    quantization: str | None = None
    trust_remote_code: bool = False
    enforce_eager: bool = False
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()

from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///./agent.db"
    WORK_DIR: str = ".workspaces"
    LLM_PROVIDER: Literal["openrouter","openai","none"] = "openrouter"
    LLM_MODEL_ID: str = "qwen/Qwen2.5-7B-Instruct"
    OPENAI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    GITHUB_TOKEN: str | None = None
    ENV: str = "dev"

    class Config:
        env_file = ".env"

settings = Settings()

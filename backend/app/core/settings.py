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
    
    # Repository cleanup settings - NEW WORKFLOW
    AUTO_CLEANUP_AFTER_REVIEW: bool = False  # Changed: repos kept until new work requested
    AUTO_CLEANUP_OLD_REPOS: bool = True      # Age-based cleanup still available
    CLEANUP_MAX_AGE_HOURS: int = 24          # Increased: repos kept longer
    CLEANUP_SAME_URL_REPOS: bool = True      # Cleanup previous repo when starting new work

    class Config:
        env_file = ".env"

settings = Settings()

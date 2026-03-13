from pathlib import Path
from pydantic_settings import BaseSettings

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

class Settings(BaseSettings):
    # Azure OpenAI — LLM
    AZURE_OPENAI_LLM_KEY: str = ""
    AZURE_LLM_ENDPOINT: str = ""
    AZURE_LLM_DEPLOYMENT_41_MINI: str = "gpt-4.1-mini"

    # Azure OpenAI — Embeddings
    AZURE_OPENAI_EMB_KEY: str = ""
    AZURE_EMB_ENDPOINT: str = ""
    AZURE_EMB_API_VERSION: str = ""
    AZURE_EMB_DEPLOYMENT: str = "text-embedding-3-large"

    # Notion
    NOTION_API_KEY: str = ""
    NOTION_DATABASE_ID: str = ""

    # PostgreSQL
    DATABASE_URL: str = "postgresql://postgres@localhost:5432/docforge_db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # App
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = str(ENV_FILE)

settings = Settings()

print(f"Loading .env from: {ENV_FILE}")
print(f"NOTION_API_KEY loaded:        {'YES' if settings.NOTION_API_KEY else 'NO'}")
print(f"AZURE_OPENAI_LLM_KEY loaded:  {'YES' if settings.AZURE_OPENAI_LLM_KEY else 'NO'}")
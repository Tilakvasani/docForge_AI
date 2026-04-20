"""
Centralized application settings for DocForge AI.

All configuration values are loaded from a `.env` file or environment variables
using Pydantic Settings. Secrets should never be committed to source control.

Required environment variables:
    AZURE_LLM_ENDPOINT           — Azure OpenAI chat endpoint URL
    AZURE_OPENAI_LLM_KEY         — Azure OpenAI API key (LLM)
    AZURE_LLM_DEPLOYMENT_41_MINI — Azure deployment name for the chat model
    AZURE_EMB_ENDPOINT           — Azure OpenAI embeddings endpoint URL
    AZURE_OPENAI_EMB_KEY         — Azure OpenAI API key (embeddings)
    AZURE_EMB_DEPLOYMENT         — Azure deployment name for the embedding model
    AZURE_EMB_API_VERSION        — Embeddings API version string
    NOTION_TOKEN                 — Notion integration token
    NOTION_DATABASE_ID           — Source document database for RAG ingest
    NOTION_TICKET_DB_ID          — Ticket tracking database for the agent layer
    CHROMA_PATH                  — Absolute path to ChromaDB persistent storage
    REDIS_URL                    — Redis connection URL
    DATABASE_URL                 — PostgreSQL connection string
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path
import os


class Settings(BaseSettings):
    """
    Application settings loaded from the environment or a `.env` file.

    Centralizes all credentials and runtime configuration for Azure OpenAI,
    Notion, Redis, ChromaDB, PostgreSQL, and general app behavior.
    Fields with empty defaults are required and must be set in `.env`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    AZURE_LLM_ENDPOINT:            str = ""
    AZURE_OPENAI_LLM_KEY:          str = ""
    AZURE_LLM_DEPLOYMENT_41_MINI:  str = "gpt-4.1-mini"
    AZURE_LLM_API_VERSION:         str = "2024-12-01-preview"

    AZURE_EMB_ENDPOINT:            str = ""
    AZURE_OPENAI_EMB_KEY:          str = ""
    AZURE_EMB_DEPLOYMENT:          str = "text-embedding-3-large"
    AZURE_EMB_API_VERSION:         str = "2024-02-01"

    NOTION_TOKEN:                  str = ""
    NOTION_API_KEY:                str = ""
    NOTION_DATABASE_ID:            str = ""
    NOTION_TICKET_DB_ID:           Optional[str] = None

    CHROMA_PATH:                   str = ""

    REDIS_URL:                     str = "redis://localhost:6379/0"

    DATABASE_URL:                  str = "postgresql://user:pass@localhost:5432/docforge"

    APP_ENV:                       str = "development"
    LOG_LEVEL:                     str = "INFO"
    CORS_ALLOWED_ORIGINS:          str = "http://localhost:8501"

    def model_post_init(self, __context):
        """
        Pydantic v2 post-initialization hook.

        Resolves `CHROMA_PATH` to an absolute filesystem path and creates
        the directory if it does not already exist.
        """
        if not self.CHROMA_PATH:
            self.CHROMA_PATH = str(Path(__file__).parent.parent.parent / "chroma_db")
        else:
            self.CHROMA_PATH = str(Path(self.CHROMA_PATH).resolve())
        Path(self.CHROMA_PATH).mkdir(parents=True, exist_ok=True)


settings = Settings()
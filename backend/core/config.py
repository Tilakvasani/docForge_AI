from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Groq
    GROQ_API_KEY: str = ""

    # Notion
    NOTION_API_KEY: str = ""
    NOTION_DATABASE_ID: str = ""

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # App
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()

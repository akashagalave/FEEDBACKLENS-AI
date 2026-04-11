from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    openai_api_key: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    redis_host: str = "localhost"
    redis_port: int = 6379
    collection_name: str = "feedbacklens"
    host: str = "0.0.0.0"
    port: int = 8003

    class Config:
        env_file = ".env"


settings = Settings()




















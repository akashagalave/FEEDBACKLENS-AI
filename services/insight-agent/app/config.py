from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    redis_host: str = "localhost"
    redis_port: int = 6379
    host: str = "0.0.0.0"
    port: int = 8003

    class Config:
        env_file = ".env"


settings = Settings()
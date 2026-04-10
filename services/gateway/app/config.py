from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    orchestrator_url: str = "http://localhost:8001"
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()
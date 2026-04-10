from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    host: str = "0.0.0.0"
    port: int = 8004

    class Config:
        env_file = ".env"


settings = Settings()
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    understanding_agent_url: str = "http://localhost:8002"
    insight_agent_url: str = "http://localhost:8003"
    recommendation_agent_url: str = "http://localhost:8004"
    host: str = "0.0.0.0"
    port: int = 8001

    class Config:
        env_file = ".env"


settings = Settings()
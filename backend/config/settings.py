
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "screenshot-to-frontend"
    app_env: str = "development"
    debug: bool = True

    llm_provider: str = "openai"
    llm_model: str
    openai_api_key: str | None = None

    aws_region: str = "ap-south-1"
    bedrock_model_id: str | None = None

    max_iterations: int = 5
    similarity_threshold: float = 0.90

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

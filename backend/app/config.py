"""Application configuration."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App settings from env."""

    openai_api_key: str = ""
    # Optional: use Anthropic instead
    anthropic_api_key: str = ""
    llm_provider: str = "openai"  # "openai" | "anthropic"

    model_query_analysis: str = "gpt-4o-mini"  # fast, cheap for classification

    class Config:
        env_file = ".env"
        extra = "ignore"

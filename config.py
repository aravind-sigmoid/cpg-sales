"""Central configuration loaded from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "cpg_sales"
    postgres_user: str = "cpg_user"
    postgres_password: str = "cpg_password"

    # OpenAI
    openai_api_key: str = "sk-dummy-replace-with-your-actual-openai-key"
    openai_model: str = "gpt-4o-mini"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # UI
    api_base_url: str = "http://localhost:8000"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )



settings = Settings()

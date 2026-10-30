from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Backend settings, read from environment variables or a backend-only .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_prefix="PORTFOLIOLAB_", extra="ignore"
    )

    # Bound to localhost by default; never expose publicly without review.
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    results_dir: Path = PROJECT_ROOT / "results"
    data_dir: Path = PROJECT_ROOT / "data" / "portfoliolab"


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Application settings, loaded from environment variables (and an optional .env file)."""

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PLACEHOLDER = "replace-me"
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
DEFAULT_FIXTURES_DIR = _REPO_ROOT / "evals" / "fixtures"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_REPO_ROOT / ".env", _BACKEND_DIR / ".env"),
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    app_base_url: str
    database_url: str  # SQLAlchemy form: postgresql+psycopg://...
    checkpoint_database_url: str  # plain postgresql://... for psycopg / LangGraph

    owner_email: str
    owner_aliases: Annotated[list[str], NoDecode] = []
    owner_passphrase: SecretStr
    session_secret: SecretStr

    anthropic_api_key: SecretStr | None = None
    llm_provider: Literal["anthropic", "scripted"] = "anthropic"
    model_triage: str = "claude-opus-5-5"
    model_plan: str = "claude-opus-5-5"
    model_draft: str = "claude-opus-5-5"
    effort_triage: str = "low"
    effort_plan: str = "medium"
    effort_draft: str = "medium"

    max_agent_steps: int = 10
    daily_budget_usd: Decimal = Decimal("2.00")
    outbound_execution_enabled: bool = False
    fixtures_dir: Path = DEFAULT_FIXTURES_DIR
    poll_interval_seconds: int = 60
    worker_lease_seconds: int = 60

    @field_validator("owner_aliases", mode="before")
    @classmethod
    def _split_aliases(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @model_validator(mode="after")
    def _reject_placeholder_secrets(self) -> "Settings":
        if not self.outbound_execution_enabled:
            return self
        secrets = {
            "session_secret": self.session_secret,
            "owner_passphrase": self.owner_passphrase,
            "anthropic_api_key": self.anthropic_api_key,
        }
        for name, secret in secrets.items():
            if secret is not None and secret.get_secret_value() == PLACEHOLDER:
                raise ValueError(
                    f"{name} is still the placeholder value; "
                    "set a real one before enabling outbound execution"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

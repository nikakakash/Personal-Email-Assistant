"""make_embeddings_client picks the implementation from settings and refuses bad configs."""

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.retrieval.embeddings import (
    FastEmbedClient,
    HashEmbeddingsClient,
    VoyageEmbeddingsClient,
    make_embeddings_client,
)


def _settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[arg-type]


def test_factory_hash(base_env: None, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
    client = make_embeddings_client(_settings(embeddings_provider="hash"), sessionmaker)
    assert isinstance(client, HashEmbeddingsClient)


def test_factory_fastembed_is_lazy(
    base_env: None, sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    client = make_embeddings_client(_settings(), sessionmaker)  # must not download anything
    assert isinstance(client, FastEmbedClient)
    assert (client.model, client.dim) == ("bge-small-en-v1.5", 384)


def test_factory_voyage(base_env: None, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
    client = make_embeddings_client(
        _settings(
            embeddings_provider="voyage",
            embed_model="voyage-3.5-lite",
            voyage_api_key=SecretStr("k"),
        ),
        sessionmaker,
    )
    assert isinstance(client, VoyageEmbeddingsClient)
    assert client.dim == 512


def test_factory_voyage_requires_key(
    base_env: None, sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        make_embeddings_client(
            _settings(embeddings_provider="voyage", embed_model="voyage-3.5-lite"), sessionmaker
        )


@pytest.mark.parametrize(
    ("provider", "model"),
    [("fastembed", "voyage-3.5-lite"), ("voyage", "bge-small-en-v1.5"), ("fastembed", "nope")],
)
def test_factory_rejects_model_provider_mismatch(
    base_env: None, sessionmaker: async_sessionmaker[AsyncSession], provider: str, model: str
) -> None:
    with pytest.raises(ValueError, match="EMBED_MODEL"):
        make_embeddings_client(
            _settings(
                embeddings_provider=provider, embed_model=model, voyage_api_key=SecretStr("k")
            ),
            sessionmaker,
        )

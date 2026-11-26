"""Async SQLAlchemy engine and session factory built from Settings."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings


def make_sessionmaker(settings: Settings) -> async_sessionmaker[AsyncSession]:
    """Build a sessionmaker on `settings.database_url`.

    `expire_on_commit=False` so rows stay readable after commit without a lazy reload
    (which would need an await in async code).
    """
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    return async_sessionmaker(engine, expire_on_commit=False)

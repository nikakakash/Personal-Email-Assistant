"""Alembic environment: runs migrations on a sync engine built from the configured URL.

The URL is `sqlalchemy.url` if the caller set it (the test harness does), otherwise
`Settings.database_url`. Migrations are hand-written, so there is no target metadata.
"""

from alembic import context
from sqlalchemy import create_engine, pool

config = context.config


def _database_url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url
    from app.config import get_settings

    return get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(url=_database_url(), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

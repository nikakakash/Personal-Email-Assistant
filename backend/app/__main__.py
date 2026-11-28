"""Command-line entry point: `python -m app api|worker|migrate|reindex`."""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from types import FrameType


def _migrate() -> None:
    """Alembic to head, then the checkpoint tables and the `app_flags` row (idempotent)."""
    from alembic import command as alembic_command
    from alembic.config import Config

    from app.config import get_settings
    from app.jobs.worker import prepare_storage

    backend_dir = Path(__file__).resolve().parent.parent
    cfg = Config(str(backend_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend_dir / "migrations"))
    alembic_command.upgrade(cfg, "head")
    asyncio.run(prepare_storage(get_settings()))


async def _reindex(args: list[str]) -> int:
    """(Re)build the retrieval index from stored mail with the configured embedding model."""
    import argparse

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.config import get_settings
    from app.llm.factory import make_llm_client
    from app.retrieval.contextualize import LLMContextualizer
    from app.retrieval.embeddings import make_embeddings_client
    from app.retrieval.indexer import reindex_all

    parser = argparse.ArgumentParser(prog="python -m app reindex")
    parser.add_argument("--account", help="only this account id")
    parser.add_argument("--force", action="store_true", help="re-embed everything")
    opts = parser.parse_args(args)
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    try:
        maker = async_sessionmaker(engine, expire_on_commit=False)
        embedder = make_embeddings_client(settings, maker)
        contextualizer = (
            LLMContextualizer(make_llm_client(settings, maker))
            if settings.rag_contextualize
            else None
        )
        stats = await reindex_all(
            maker,
            embedder,
            account_id=opts.account,
            force=opts.force,
            contextualizer=contextualizer,
        )
    finally:
        await engine.dispose()
    print(
        f"reindex ({embedder.model}): {stats.messages} messages, {stats.chunks} chunks, "
        f"{stats.embedded} embedded"
    )
    return 0


async def _worker() -> None:
    """Run the worker until SIGINT/SIGTERM; the job in progress finishes first."""
    from app.config import get_settings
    from app.jobs.worker import run_worker

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # Windows event loops have no signal handlers. A plain handler for Ctrl+C asks
            # for a graceful stop; a second Ctrl+C raises KeyboardInterrupt and exits now.
            def on_interrupt(signum: int, frame: FrameType | None) -> None:
                if stop.is_set():
                    raise KeyboardInterrupt
                loop.call_soon_threadsafe(stop.set)

            if sig == signal.SIGINT:
                signal.signal(sig, on_interrupt)
    await run_worker(get_settings(), stop)


def main(argv: list[str]) -> int:
    if sys.platform == "win32":
        # psycopg async needs a selector event loop on Windows.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    command = argv[0] if argv else ""
    if command == "api":
        import uvicorn

        from app.config import get_settings

        settings = get_settings()
        uvicorn.run(
            "app.main:create_app", factory=True, host=settings.api_host, port=settings.api_port
        )
        return 0
    if command == "migrate":
        _migrate()
        return 0
    if command == "worker":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        try:
            asyncio.run(_worker())
        except KeyboardInterrupt:
            return 130
        return 0
    if command == "reindex":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        return asyncio.run(_reindex(argv[1:]))
    print("usage: python -m app {api|worker|migrate|reindex}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

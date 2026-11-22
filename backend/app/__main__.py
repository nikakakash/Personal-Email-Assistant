"""Command-line entry point: `python -m app api|worker|migrate`."""

import asyncio
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if sys.platform == "win32":
        # psycopg async needs a selector event loop on Windows.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    command = argv[0] if argv else ""
    if command == "api":
        import uvicorn

        uvicorn.run("app.main:create_app", factory=True, host="127.0.0.1", port=8000)
        return 0
    if command == "migrate":
        from alembic import command as alembic_command
        from alembic.config import Config

        backend_dir = Path(__file__).resolve().parent.parent
        cfg = Config(str(backend_dir / "alembic.ini"))
        cfg.set_main_option("script_location", str(backend_dir / "migrations"))
        alembic_command.upgrade(cfg, "head")  # checkpointer setup is added in Task 11
        return 0
    if command == "worker":
        print(f"{command}: not implemented yet", file=sys.stderr)
        return 1
    print("usage: python -m app {api|worker|migrate}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

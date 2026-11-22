"""Command-line entry point: `python -m app api|worker|migrate`."""

import asyncio
import sys


def main(argv: list[str]) -> int:
    if sys.platform == "win32":
        # psycopg async needs a selector event loop on Windows.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    command = argv[0] if argv else ""
    if command == "api":
        import uvicorn

        uvicorn.run("app.main:create_app", factory=True, host="127.0.0.1", port=8000)
        return 0
    if command in ("worker", "migrate"):
        print(f"{command}: not implemented yet", file=sys.stderr)
        return 1
    print("usage: python -m app {api|worker|migrate}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

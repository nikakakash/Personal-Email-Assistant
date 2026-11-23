# Backend image: API, worker and migrations share it (`python -m app api|worker|migrate`).
# Build context is the repository root, so the eval/E2E fixtures under evals/ are included.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    API_HOST=0.0.0.0

WORKDIR /app/backend

# Dependencies first (cached unless the lockfile changes), then the code.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY backend/ ./
COPY evals/ /app/evals/
RUN uv sync --locked --no-dev

RUN useradd --system --uid 10001 --no-create-home app
USER app

EXPOSE 8000
CMD ["python", "-m", "app", "api"]

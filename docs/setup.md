# Setup

This milestone runs entirely on mock mail and calendar providers: nothing connects to a
real mailbox, and "sending" writes a row to the mock outbox. The only external service is
the Anthropic API, used for triage, planning and drafting.

## Prerequisites

| Tool | Version | Used for |
|---|---|---|
| Docker Desktop (or Docker Engine + Compose v2.24+) | current | the full stack, Postgres for tests |
| [uv](https://docs.astral.sh/uv/) | 0.13+ | backend dependencies, tests, the eval runner |
| Node.js | current LTS (26.x) | frontend dev server, unit tests, Playwright |

On Windows, run commands from PowerShell or Git Bash. The backend already selects the
event loop psycopg needs on Windows.

## 1. Configure `.env`

```bash
cp .env.example .env
```

Then edit at least these:

| Variable | What to set |
|---|---|
| `OWNER_EMAIL` | the address the assistant acts for (mail from it never starts a workflow) |
| `OWNER_ALIASES` | other addresses that are also you, comma-separated |
| `OWNER_PASSPHRASE` | the login passphrase, **12+ characters** (login answers 503 until it is set) |
| `SESSION_SECRET` | a random string, 12+ characters, e.g. `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `ANTHROPIC_API_KEY` | see below |
| `DAILY_BUDGET_USD` | the hard daily ceiling for model spend (default `2.00`) |
| `OUTBOUND_EXECUTION_ENABLED` | leave `false` until you want approved mock sends to execute |

Compose overrides `DATABASE_URL` and `CHECKPOINT_DATABASE_URL` for the containers; the
values in `.env` are for running the backend directly on your machine. Keep
`APP_BASE_URL=http://localhost:8080`: it is the origin the browser uses in both setups, and
sign-in is refused (`csrf_failed`) from any other origin.

### Getting an Anthropic API key

Either:

- create a key in the [Claude Console](https://console.anthropic.com/) under **API keys** and
  put it in `.env` as `ANTHROPIC_API_KEY=...`, or
- with the `ant` CLI, run `ant auth login`, then copy the key it gives you into `.env`.

The key stays in the backend. It never reaches the browser, graph state or traces. To try
the app without a key, set `LLM_PROVIDER=scripted`. The model is then replaced by the canned
outputs stored in the fixtures, at zero cost. Only fixtures that carry a `script` do
anything (see `evals/e2e-fixtures/`).

## 2. Run the stack

```bash
docker compose up -d --build
```

| Service | What it does |
|---|---|
| `db` | Postgres 17.6 (published on `127.0.0.1:5433` for tests and local tools) |
| `migrate` | one-shot: Alembic migrations, LangGraph checkpoint tables, the flags row |
| `api` | FastAPI on port 8000 (internal) |
| `worker` | runs workflows from the job queue; restarts automatically |
| `web` | Caddy on **http://localhost:8080**: the UI, with `/api/*` proxied to `api` |

Check it with `curl http://localhost:8080/api/v1/health/ready`, which should return
`{"status":"ok"}`. Then open http://localhost:8080 and sign in with `OWNER_PASSPHRASE`.

To try a workflow, go to **Settings → Developer tools**, pick a fixture (for example
`question-lunch`) and click **Inject fixture**. The email shows up in the queue once the
worker has triaged it.

Stop with `docker compose down`. Add `-v` to also delete the database.

## 3. Develop without containers

```bash
docker compose up -d db
cd backend
uv sync
uv run python -m app migrate
uv run python -m app api        # terminal 1: http://127.0.0.1:8000
uv run python -m app worker     # terminal 2
cd ../frontend
npm ci
npm run dev                     # terminal 3: http://localhost:8080 (proxies /api)
```

## 4. Tests

```bash
cd backend && uv run pytest -q                     # needs `docker compose up -d db`
uv run ruff check . ../evals && uv run mypy app
cd ../frontend && npm run test -- --run && npm run build
```

### End-to-end journeys

The E2E overlay runs a separate Compose project with its own fresh database. It uses the
scripted model, the E2E fixtures, and outbound execution against the mock providers. The UI
is served on port 8081, so it can run next to the dev stack.

```bash
docker compose -p email-assistant-e2e -f compose.yaml -f compose.e2e.yaml up -d --build
cd frontend
npx playwright install chromium      # first time only
npx playwright test
cd ..
docker compose -p email-assistant-e2e -f compose.yaml -f compose.e2e.yaml down -v
```

Each journey injects its own fixture, and fixtures cannot be injected twice. Always run
`down -v` before re-running the suite.

## 5. Triage evaluation (costs money)

```bash
cd backend
uv run python ../evals/run_triage_eval.py --split heldout          # prints the estimate only
uv run python ../evals/run_triage_eval.py --split all --yes         # calls Claude
```

The runner uses the real Anthropic client, and its spend counts toward `DAILY_BUDGET_USD`.
It prints accuracy, recall of "respond" (target ≥ 0.95), a confusion matrix and the actual
cost, and writes `evals/results/<date>-<model>.json`.

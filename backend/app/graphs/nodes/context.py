"""`load_context` and `load_preferences`: the first two nodes of the parent graph.

Both are deterministic reads. Each returns a *partial update*: only the keys it sets
(`thread`, `source_version` / `preferences`, `preference_version`); LangGraph merges them into
the state and checkpoints the result. Providers and the DB sessionmaker come from the runtime
context (`runtime.context`, a `GraphDeps`), never from state.

`source_version` is the thread version the whole workflow reasons about. Preflight compares it
with a fresh read before anything is sent, so a thread that changed since is caught.
"""

import asyncio
from typing import Any

from langgraph.runtime import Runtime
from sqlalchemy import select

from app.db.models import PreferenceVersion
from app.graphs.deps import GraphDeps
from app.graphs.nodes.reads import THREAD_MESSAGE_LIMIT, TOOL_TIMEOUT_SECONDS
from app.graphs.state import EmailState


async def load_context(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Load the thread snapshot. A provider failure propagates (the worker retries the run)."""
    snapshot = await asyncio.wait_for(
        runtime.context.mail.get_thread(state["provider_thread_id"], limit=THREAD_MESSAGE_LIMIT),
        TOOL_TIMEOUT_SECONDS,
    )
    return {"thread": snapshot.model_dump(mode="json"), "source_version": snapshot.version}


async def load_preferences(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Active preferences per namespace (highest active revision wins).

    `preference_version` is the highest active revision across namespaces, 0 when none exist.
    The app has a single owner, so rows are not filtered by owner.
    """
    async with runtime.context.sessionmaker() as session:
        rows = (
            await session.execute(
                select(
                    PreferenceVersion.namespace, PreferenceVersion.revision, PreferenceVersion.value
                )
                .where(PreferenceVersion.status == "active")
                .order_by(PreferenceVersion.namespace, PreferenceVersion.revision.desc())
            )
        ).all()
    preferences: dict[str, Any] = {}
    version = 0
    for namespace, revision, value in rows:
        if namespace not in preferences:  # rows are sorted newest first per namespace
            preferences[namespace] = value
            version = max(version, revision)
    return {"preferences": preferences, "preference_version": version}

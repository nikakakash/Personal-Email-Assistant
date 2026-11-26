"""Contextual retrieval: a model writes one sentence situating each chunk within its email.

A chunk like "Yes, 5,200 works." is hard to find on its own; "Owner accepts Brightweb's revised
website quote" makes it findable by meaning and by keyword. The sentence is stored as
`message_chunks.context_prefix` and indexed (embedding text and tsvector) only. Search
returns the chunk text, never the prefix, so model-written text never reaches the drafting
prompt, and an email that tries to steer its own summary can at most change how it is found.

The whole email goes in `cached_context` (identical for every chunk of the message, so the
provider's prompt cache reuses it), and the chunk goes in the user turn. Spend goes through the
budget guard like every model call (purpose "contextualize").
"""

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.llm.client import LLMCall, LLMClient
from app.prompts import load_prompt

MESSAGE_MAX_CHARS = 6000


class ChunkContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context: str = Field(
        max_length=300,
        description="One sentence: who wrote to whom, about what, and the key fact of the chunk.",
    )


@dataclass(frozen=True)
class MessageInfo:
    provider_message_id: str
    sender: str
    to: list[str]
    subject: str
    date: str
    body: str


class Contextualizer(Protocol):
    async def __call__(self, message: MessageInfo, chunk_text: str) -> str: ...


class LLMContextualizer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def __call__(self, message: MessageInfo, chunk_text: str) -> str:
        prompt = load_prompt("contextualize")
        email = (
            "<email>\n"
            f"From: {message.sender}\nTo: {', '.join(message.to)}\nDate: {message.date}\n"
            f"Subject: {message.subject}\n\n{message.body[:MESSAGE_MAX_CHARS]}\n</email>"
        )
        result = await self._llm.structured(
            LLMCall(
                purpose="contextualize",
                system=prompt.system,
                cached_context=email,
                user_content=f"<chunk>\n{chunk_text}\n</chunk>",
                output_model=ChunkContext,
                script_key=message.provider_message_id,
                max_tokens=200,
            )
        )
        return " ".join(result.output.context.split())

"""Versioned prompt templates stored as `<name>.md` with `version:` front matter.

Each file starts with a `---` block containing `version: <name>_v1`; the rest is the system
prompt. The version is recorded with every model run so behavior changes can be traced.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

_DIR = Path(__file__).parent

PROMPT_VERSIONS: dict[str, str] = {
    "triage": "triage_v1",
    "plan": "plan_v2",
    "draft_reply": "draft_reply_v2",
    "draft_meeting": "draft_meeting_v2",
}


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    system: str  # prompt body without front matter
    text: str  # the whole file, including front matter


def _parse(name: str, text: str) -> Prompt:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"prompt {name} lacks front matter")
    end = next((i for i, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), None)
    if end is None:
        raise ValueError(f"prompt {name} has unterminated front matter")
    meta: dict[str, str] = {}
    for ln in lines[1:end]:
        key, sep, value = ln.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    version = meta.get("version")
    if version is None:
        raise ValueError(f"prompt {name} lacks a version")
    return Prompt(name, version, "\n".join(lines[end + 1 :]).strip(), text)


@cache
def load_prompt(name: str) -> Prompt:
    if name not in PROMPT_VERSIONS:
        raise KeyError(f"unknown prompt {name!r}")
    prompt = _parse(name, (_DIR / f"{name}.md").read_text(encoding="utf-8"))
    if prompt.version != PROMPT_VERSIONS[name]:
        raise ValueError(f"prompt {name} version {prompt.version} != {PROMPT_VERSIONS[name]}")
    return prompt

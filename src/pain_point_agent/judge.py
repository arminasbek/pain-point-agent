from typing import NamedTuple

from anthropic.types import ToolParam, ToolUseBlock

from pain_point_agent.client import LLMClient
from pain_point_agent.config import settings
from pain_point_agent.prompts import load_prompt

JUDGE_PROMPT = load_prompt("judge")
FIT_PROMPT = load_prompt("fit")

VERDICT_TOOL: ToolParam = {
    "name": "verdict",
    "strict": True,
    "description": "Record whether the quote is a complaint.",
    "input_schema": {
        "type": "object",
        "properties": {
            "complaint": {"type": "boolean"},
            "reason": {"type": "string", "description": "One short sentence"},
        },
        "required": ["complaint", "reason"],
        "additionalProperties": False,
    },
}

FIT_TOOL: ToolParam = {
    "name": "fit",
    "strict": True,
    "description": "Record whether the quote is evidence for the named problem.",
    "input_schema": {
        "type": "object",
        "properties": {
            "fits": {"type": "boolean"},
            "reason": {"type": "string", "description": "One short sentence"},
        },
        "required": ["fits", "reason"],
        "additionalProperties": False,
    },
}


class Verdict(NamedTuple):
    complaint: bool
    reason: str


class Fit(NamedTuple):
    fits: bool
    reason: str


class JudgeError(RuntimeError):
    pass


def judge_quote(quote: str, post: str, client: LLMClient) -> Verdict:
    return Verdict(*_ask(JUDGE_PROMPT, VERDICT_TOOL, "complaint", f"Quote:\n{quote}\n\nPost:\n{post}", client))


def in_thread(url: str, text: str) -> str:
    return f"Thread: {url}\n\n{text}"


def judge_fit(pain: str, quote: str, post: str, client: LLMClient, niche: str = "") -> Fit:
    content = f"Problem:\n{pain}\n\nQuote:\n{quote}\n\nPost:\n{post}"
    if niche:
        content = f"Niche:\n{niche}\n\n{content}"
    return Fit(*_ask(FIT_PROMPT, FIT_TOOL, "fits", content, client))


def _ask(system: str, tool: ToolParam, field: str, content: str, client: LLMClient) -> tuple[bool, str]:
    response = client.messages.create(
        model=settings.model,
        max_tokens=settings.judge_max_tokens,
        output_config={"effort": settings.judge_effort},
        timeout=settings.judge_timeout,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": content}],
    )
    block = next((block for block in response.content if isinstance(block, ToolUseBlock)), None)
    value = block.input.get(field) if block else None
    if block is None or not isinstance(value, bool):
        raise JudgeError(f"Judge gave no verdict (stop_reason={response.stop_reason})")
    return value, str(block.input.get("reason", ""))

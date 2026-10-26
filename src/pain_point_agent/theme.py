from anthropic.types import TextBlock

from pain_point_agent.client import LLMClient
from pain_point_agent.config import settings
from pain_point_agent.prompts import load_prompt
from pain_point_agent.research import Evidence

THEME_PROMPT = load_prompt("theme")


class ThemeError(RuntimeError):
    pass


def format_pains(niche: str, pains: dict[str, list[Evidence]]) -> str:
    lines = [f"Niche: {niche}"]
    for number, (pain, evidence) in enumerate(pains.items(), 1):
        lines += ["", f"{number}. {pain}"]
        lines += [f'- "{item["quote"]}"' for item in evidence]
    return "\n".join(lines)


def find_theme(niche: str, pains: dict[str, list[Evidence]], client: LLMClient) -> str:
    response = client.messages.create(
        model=settings.model,
        max_tokens=settings.theme_max_tokens,
        system=THEME_PROMPT,
        messages=[{"role": "user", "content": format_pains(niche, pains)}],
    )
    if response.stop_reason != "end_turn":
        raise ThemeError(f"Theme call stopped with stop_reason={response.stop_reason}")
    return "".join(block.text for block in response.content if isinstance(block, TextBlock)).strip()

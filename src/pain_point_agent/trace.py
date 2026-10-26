import json
import sys
from pathlib import Path
from typing import Any

from anthropic.types import Message

from pain_point_agent.client import LLMClient
from pain_point_agent.config import settings


def show(line: str) -> None:
    print(line, file=sys.stderr)


def cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def summarize(content: str) -> str:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return cut(content, 160)
    if isinstance(data, list):
        return f"{len(data)} results"
    if isinstance(data, dict) and "posts" in data:
        return f'thread "{cut(data["title"], 60)}" ({len(data["posts"])} posts)'
    return cut(content, 160)


class Trace:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.rounds = 0
        self.input_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.output_tokens = 0
        self.check_calls = 0
        self.check_input_tokens = 0
        self.check_output_tokens = 0

    def round(self, sent: int, usage: Any) -> None:
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        self.rounds += 1
        self.input_tokens += usage.input_tokens
        self.cache_read_tokens += cache_read
        self.cache_write_tokens += cache_write
        self.output_tokens += usage.output_tokens
        if self.verbose:
            show(
                f"\n── round {self.rounds} ── sent {sent} messages · "
                f"{usage.input_tokens:,} input tokens · {cache_read:,} from cache · {cache_write:,} written to cache · "
                f"{usage.output_tokens:,} output tokens"
            )

    def check(self, usage: Any) -> None:
        self.check_calls += 1
        self.check_input_tokens += (
            usage.input_tokens + (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0)
        )
        self.check_output_tokens += usage.output_tokens

    def cost(self) -> float:
        loop_input = (
            self.input_tokens
            + settings.cache_read_rate * self.cache_read_tokens
            + settings.cache_write_rate * self.cache_write_tokens
        )
        input_cost = (loop_input + self.check_input_tokens) * settings.price_input_per_mtok
        output_cost = (self.output_tokens + self.check_output_tokens) * settings.price_output_per_mtok
        return (input_cost + output_cost) / 1_000_000

    def summary(self) -> dict[str, float]:
        return {
            "rounds": self.rounds,
            "loop_input_tokens": self.input_tokens,
            "loop_cache_read_tokens": self.cache_read_tokens,
            "loop_cache_write_tokens": self.cache_write_tokens,
            "loop_output_tokens": self.output_tokens,
            "check_calls": self.check_calls,
            "check_input_tokens": self.check_input_tokens,
            "check_output_tokens": self.check_output_tokens,
            "cost_usd": self.cost(),
        }

    def call(self, name: str, tool_input: dict[str, Any]) -> None:
        if self.verbose:
            args = ", ".join(f"{key}={value!r}" for key, value in tool_input.items())
            show(f"Claude → {name}({cut(args, 160)})")
        else:
            show(f"-> {name}({tool_input})")

    def result(self, content: str, is_error: bool) -> None:
        if self.verbose:
            show(f"loop   ← {'REJECTED: ' + cut(content, 160) if is_error else summarize(content)}")
        elif is_error:
            show(f"   rejected: {content}")

    def totals(self) -> None:
        if self.verbose:
            show(
                f"\n── {self.rounds} rounds · {self.input_tokens:,} input tokens · "
                f"{self.cache_read_tokens:,} from cache · {self.cache_write_tokens:,} written to cache · "
                f"{self.output_tokens:,} output tokens (agent loop)"
            )
            show(
                f"── checks (judge, fit, theme): {self.check_calls} calls · {self.check_input_tokens:,} input tokens · "
                f"{self.check_output_tokens:,} output tokens"
            )
            show(f"── cost ≈ ${self.cost():.4f} at {settings.model} list prices")


class Metered:
    def __init__(self, client: LLMClient, trace: Trace) -> None:
        self._client = client
        self._trace = trace

    @property
    def messages(self) -> "Metered":
        return self

    def create(self, **kwargs: Any) -> Message:
        response = self._client.messages.create(**kwargs)
        if getattr(response, "usage", None) is not None:
            self._trace.check(response.usage)
        return response


def save(messages: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(messages, default=lambda block: block.model_dump(), ensure_ascii=False, indent=2))

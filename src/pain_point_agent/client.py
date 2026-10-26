from collections.abc import Callable
from typing import Protocol

import anthropic
from anthropic.types import Message

from pain_point_agent.config import settings


class Messages(Protocol):
    @property
    def create(self) -> Callable[..., Message]: ...


class LLMClient(Protocol):
    @property
    def messages(self) -> Messages: ...


def make_client() -> anthropic.Anthropic:
    api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    return anthropic.Anthropic(api_key=api_key)

from types import SimpleNamespace

import pytest
from anthropic.types import TextBlock

from pain_point_agent import theme
from pain_point_agent.research import Evidence

PAINS = {
    "Bot goes silent": [
        Evidence(source="n8n", title="T1", url="https://x/1", quote="every few days the bot just goes silent"),
        Evidence(source="n8n", title="T2", url="https://x/2", quote="customers say the bot never answered"),
    ]
}


class StubClient:
    def __init__(self, response):
        self.calls = []
        self.messages = SimpleNamespace(create=lambda **kwargs: self.calls.append(kwargs) or response)


def text_response(text, stop_reason="end_turn"):
    return SimpleNamespace(content=[TextBlock(type="text", text=text)], stop_reason=stop_reason)


def test_theme_call_sees_only_verified_pains_and_no_tools():
    client = StubClient(text_response(" Bots fail silently. "))

    result = theme.find_theme("WhatsApp bots", PAINS, client)

    assert result == "Bots fail silently."
    call = client.calls[0]
    assert "tools" not in call
    content = call["messages"][0]["content"]
    assert "1. Bot goes silent" in content
    assert '- "customers say the bot never answered"' in content


def test_theme_call_raises_when_cut_off():
    client = StubClient(text_response("Bots fail", stop_reason="max_tokens"))

    with pytest.raises(theme.ThemeError, match="max_tokens"):
        theme.find_theme("WhatsApp bots", PAINS, client)

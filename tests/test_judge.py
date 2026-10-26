from types import SimpleNamespace

import pytest
from anthropic.types import TextBlock, ToolUseBlock

from pain_point_agent import judge


class StubClient:
    def __init__(self, response):
        self.calls = []
        self.messages = SimpleNamespace(create=lambda **kwargs: self.calls.append(kwargs) or response)


def verdict_response(tool_input):
    block = ToolUseBlock(type="tool_use", id="toolu_1", name="verdict", input=tool_input)
    return SimpleNamespace(content=[block], stop_reason="tool_use")


def test_judge_returns_verdict_and_forces_the_verdict_tool():
    client = StubClient(verdict_response({"complaint": False, "reason": "Advice."}))

    result = judge.judge_quote("check the CT clamp direction", "Hi, check the CT clamp direction first.", client)

    assert result == judge.Verdict(False, "Advice.")
    call = client.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "verdict"}
    assert "check the CT clamp direction first." in call["messages"][0]["content"]


def test_judge_raises_without_a_verdict():
    client = StubClient(SimpleNamespace(content=[TextBlock(type="text", text="yes")], stop_reason="end_turn"))

    with pytest.raises(judge.JudgeError):
        judge.judge_quote("q", "p", client)


def test_judge_raises_on_non_boolean_verdict():
    client = StubClient(verdict_response({"complaint": "yes", "reason": "r"}))

    with pytest.raises(judge.JudgeError):
        judge.judge_quote("q", "p", client)


def fit_response(tool_input):
    block = ToolUseBlock(type="tool_use", id="toolu_1", name="fit", input=tool_input)
    return SimpleNamespace(content=[block], stop_reason="tool_use")


def test_fit_returns_verdict_and_shows_the_problem():
    client = StubClient(fit_response({"fits": False, "reason": "About pricing."}))

    result = judge.judge_fit("Bot goes silent", "the plan costs too much", "Honestly the plan costs too much.", client)

    assert result == judge.Fit(False, "About pricing.")
    call = client.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "fit"}
    content = call["messages"][0]["content"]
    assert "Problem:\nBot goes silent" in content
    assert "Honestly the plan costs too much." in content


def test_fit_shows_the_niche():
    client = StubClient(fit_response({"fits": False, "reason": "About the OpenAI Agents SDK, not n8n."}))

    judge.judge_fit("Agent hits max turns", "q", "post", client, niche="n8n + AI agents")

    assert client.calls[0]["messages"][0]["content"].startswith("Niche:\nn8n + AI agents\n\nProblem:")


def test_fit_raises_without_a_verdict():
    client = StubClient(SimpleNamespace(content=[TextBlock(type="text", text="yes")], stop_reason="end_turn"))

    with pytest.raises(judge.JudgeError):
        judge.judge_fit("p", "q", "post", client)

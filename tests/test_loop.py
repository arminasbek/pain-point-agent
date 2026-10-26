import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from anthropic.types import TextBlock, ToolUseBlock
from sqlalchemy import text

from pain_point_agent import loop, sources, tools
from pain_point_agent.config import settings
from pain_point_agent.db import Database
from pain_point_agent.judge import Fit, Verdict
from pain_point_agent.research import Research


class StubClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


USAGE = SimpleNamespace(input_tokens=1200, output_tokens=80, cache_read_input_tokens=None, cache_creation_input_tokens=None)
CACHED_USAGE = SimpleNamespace(input_tokens=50, output_tokens=80, cache_read_input_tokens=1000, cache_creation_input_tokens=150)


def tool_use_response(name, tool_input, usage=USAGE):
    block = ToolUseBlock(type="tool_use", id="toolu_1", name=name, input=tool_input)
    return SimpleNamespace(content=[block], stop_reason="tool_use", usage=usage)


def report(niche, client, **kwargs):
    return loop.run_research(niche, client, **kwargs).report(niche)


def search_response(query):
    return tool_use_response("search", {"source": "n8n", "query": query})


@pytest.fixture
def stub_search(monkeypatch):
    queries = []

    def search(source, query):
        queries.append((source, query))
        return [{"id": 1, "title": "Bot goes silent"}]

    monkeypatch.setattr(sources, "search", search)
    return queries


def test_loop_runs_tool_and_returns_report_after_finish(stub_search, monkeypatch):
    monkeypatch.setattr(settings, "min_pains", 0)
    client = StubClient([search_response("whatsapp bot"), tool_use_response("finish", {"note": "nothing found"})])

    text = report("WhatsApp bots", client)

    assert text.startswith("# Pain report: WhatsApp bots")
    assert "Agent's note (written by the model, not checked by code): nothing found" in text
    assert stub_search == [("n8n", "whatsapp bot")]
    tool_result = client.calls[1]["messages"][2]["content"][0]
    assert tool_result["tool_use_id"] == "toolu_1"
    assert tool_result["is_error"] is False
    assert "Bot goes silent" in tool_result["content"]
    assert client.calls[0]["tool_choice"] == {"type": "any"}
    assert client.calls[0]["cache_control"] == {"type": "ephemeral"}


def test_rejected_finish_keeps_loop_running(stub_search):
    client = StubClient([tool_use_response("finish", {"note": ""}), search_response("q")])

    text = report("niche", client, max_steps=2)

    assert "Stopped early: step limit of 2 reached before finish" in text
    tool_result = client.calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "Not done" in tool_result["content"]


def test_cut_off_answer_returns_a_partial_report():
    response = SimpleNamespace(content=[TextBlock(type="text", text="1. Bot go")], stop_reason="max_tokens", usage=USAGE)

    text = report("niche", StubClient([response]))

    assert "Stopped early: model stopped with stop_reason=max_tokens" in text


def test_api_error_returns_a_partial_report():
    class FailingClient:
        messages = SimpleNamespace(create=None)

        def __init__(self):
            def create(**kwargs):
                raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))

            self.messages = SimpleNamespace(create=create)

    text = report("niche", FailingClient())

    assert "Stopped early: API error: Connection error." in text


def test_each_round_tells_the_model_its_budget(stub_search):
    client = StubClient([search_response("q"), search_response("q2")])

    report("niche", client, max_steps=2)

    budget = client.calls[1]["messages"][2]["content"][-1]
    assert budget["type"] == "text"
    assert budget["text"].startswith("Budget: 1 steps left · 5 searches left")


def test_run_research_returns_the_state(stub_search, monkeypatch):
    monkeypatch.setattr(settings, "min_pains", 0)
    client = StubClient([search_response("q"), tool_use_response("finish", {"note": "done"})])

    research = loop.run_research("niche", client)

    assert research.finished is True
    assert research.stopped == ""
    assert research.searches == 1
    assert research.usage["rounds"] == 2
    assert research.usage["cost_usd"] > 0


def test_trace_prints_rounds_and_saves_conversation(stub_search, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(settings, "min_pains", 0)
    client = StubClient([search_response("whatsapp bot"), tool_use_response("finish", {"note": ""}, CACHED_USAGE)])
    trace_file = tmp_path / "trace.json"

    report("WhatsApp bots", client, trace_file=trace_file)

    err = capsys.readouterr().err
    assert "── round 1 ── sent 1 messages · 1,200 input tokens" in err
    assert "── round 2 ── sent 3 messages" in err
    assert "loop   ← 1 results" in err
    assert "── round 2 ── sent 3 messages · 50 input tokens · 1,000 from cache · 150 written to cache" in err
    assert "── 2 rounds · 1,250 input tokens · 1,000 from cache · 150 written to cache" in err
    saved = json.loads(trace_file.read_text())
    assert saved[0] == {"role": "user", "content": "Niche: WhatsApp bots"}
    assert saved[1]["content"][0]["name"] == "search"


def test_loop_stops_at_max_steps(stub_search):
    client = StubClient([search_response("q")] * 3)

    research = loop.run_research("niche", client, max_steps=3)

    assert research.finished is False
    assert research.stopped == "step limit of 3 reached before finish"
    assert len(client.calls) == 3


def test_unknown_tool_returns_error():
    content, is_error, failed = tools.run_tool("nope", {}, Research(judge=lambda quote, post: Verdict(True, ""), fit=lambda pain, quote, post: Fit(True, "")))

    assert is_error is True
    assert "Unknown tool" in content


def test_extra_argument_is_rejected_before_any_request(stub_search):
    content, is_error, failed = tools.run_tool("search", {"source": "n8n", "query": "x", "base_url": "http://169.254.169.254"}, Research(judge=lambda quote, post: Verdict(True, ""), fit=lambda pain, quote, post: Fit(True, "")))

    assert is_error is True
    assert "base_url" in content
    assert stub_search == []


def test_wrong_argument_type_is_rejected():
    content, is_error, failed = tools.run_tool("fetch", {"source": "n8n", "id": "../admin"}, Research(judge=lambda quote, post: Verdict(True, ""), fit=lambda pain, quote, post: Fit(True, "")))

    assert is_error is True
    assert "must be integer" in content


def test_unknown_source_is_rejected(stub_search):
    content, is_error, failed = tools.run_tool("search", {"source": "http://localhost:8080", "query": "x"}, Research(judge=lambda quote, post: Verdict(True, ""), fit=lambda pain, quote, post: Fit(True, "")))

    assert is_error is True
    assert "must be one of" in content
    assert stub_search == []


def test_tool_schema_lists_enabled_sources_only():
    source = tools.TOOLS[0]["input_schema"]["properties"]["source"]
    enabled = [name for name, config in settings.sources.items() if config.enabled]

    assert source["enum"] == enabled
    assert all(f"{name}:" in source["description"] for name in enabled)
    assert "appstore" not in source["enum"]
    assert "appstore:" not in source["description"]


def test_report_survives_a_failed_theme_step(stub_search, monkeypatch, capsys):
    monkeypatch.setattr(settings, "min_pains", 0)
    evidence = [{"source": "n8n", "title": "T", "url": "https://community.n8n.io/t/x/1", "quote": "bot goes silent", "date": ""}]
    monkeypatch.setattr(loop.Research, "strong_pains", lambda self: {"Bot goes silent": evidence})

    def failing_theme(niche, pains, client):
        raise loop.ThemeError("Theme call stopped with stop_reason=max_tokens")

    monkeypatch.setattr(loop, "find_theme", failing_theme)
    client = StubClient([tool_use_response("finish", {"note": ""})])

    text = report("WhatsApp bots", client)

    assert "## 1. Bot goes silent" in text
    assert "Theme (inference)" not in text
    assert "theme step failed: Theme call stopped with stop_reason=max_tokens" in capsys.readouterr().err


def test_consensus_runs_several_times_and_explains_itself(stub_search, monkeypatch):
    monkeypatch.setattr(settings, "min_pains", 0)
    client = StubClient([search_response("q"), tool_use_response("finish", {"note": ""})] * 3)

    research = loop.run_consensus("niche", client, runs=3)

    assert len(client.calls) == 6
    assert research.searches == 3
    assert research.finished is True
    report = research.report("niche")
    assert "Consensus of 3 runs: a problem is kept when at least 2 of its threads were found by 2 or more runs." in report
    assert "Problems per run: 0, 0, 0." in report


def test_consensus_says_which_runs_stopped_early(stub_search, monkeypatch):
    monkeypatch.setattr(settings, "min_pains", 0)
    monkeypatch.setattr(settings, "max_steps", 2)
    finished = [search_response("q"), tool_use_response("finish", {"note": ""})]
    client = StubClient(finished + [search_response("q"), search_response("q")] + finished)

    research = loop.run_consensus("niche", client, runs=3)

    assert research.stopped == "run 2: step limit of 2 reached before finish"
    assert research.finished is False
    assert "Stopped early: run 2: step limit of 2 reached before finish." in research.report("niche")


def test_fit_judge_is_told_the_niche(monkeypatch):
    seen = []
    monkeypatch.setattr(loop, "judge_fit", lambda pain, quote, post, client, niche: seen.append(niche) or Fit(True, ""))

    research = loop.run_research("n8n + AI agents", StubClient([search_response("q")]), max_steps=1)
    research.fit("pain", "quote", "post")

    assert seen == ["n8n + AI agents"]


def test_run_is_saved_to_the_database(stub_search, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "min_pains", 0)
    db = Database(tmp_path / "pain.db")
    client = StubClient([search_response("bot"), tool_use_response("finish", {"note": "done"})])

    loop.run_research("WhatsApp bots", client, db=db)

    run = rows(db, "SELECT niche, finished, note, report FROM runs")[0]
    assert run[:3] == ("WhatsApp bots", 1, "done")
    assert run[3].startswith("# Pain report: WhatsApp bots")
    assert rows(db, "SELECT source, query FROM searches") == [("n8n", "bot")]
    db.close()


def test_consensus_saves_each_run_linked_to_the_combined_result(stub_search, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "min_pains", 0)
    db = Database(tmp_path / "pain.db")
    client = StubClient([tool_use_response("finish", {"note": ""})] * 3)

    loop.run_consensus("niche", client, runs=3, db=db)

    assert rows(db, "SELECT id, kind, parent_id, finished FROM runs ORDER BY id") == [
        (1, "consensus", None, 1),
        (2, "single", 1, 1),
        (3, "single", 1, 1),
        (4, "single", 1, 1),
    ]
    db.close()


def test_step_limit_is_saved_as_an_error(stub_search, tmp_path):
    db = Database(tmp_path / "pain.db")

    loop.run_research("niche", StubClient([search_response("q")] * 2), max_steps=2, db=db)

    assert rows(db, "SELECT stage, error FROM errors") == [("budget", "step limit of 2 reached before finish")]
    assert rows(db, "SELECT finished, errors FROM runs") == [(0, 1)]
    db.close()


def test_consensus_crash_still_finishes_the_combined_run(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "min_pains", 0)
    calls = 0

    def tool(name, tool_input, research):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyError("boom")
        return tools.run_tool(name, tool_input, research)

    monkeypatch.setattr(loop, "run_tool", tool)
    db = Database(tmp_path / "pain.db")
    client = StubClient([tool_use_response("finish", {"note": ""})] * 2)

    with pytest.raises(KeyError, match="boom"):
        loop.run_consensus("niche", client, runs=3, db=db)

    assert rows(db, "SELECT kind, finished, stopped, finished_at IS NOT NULL FROM runs ORDER BY id") == [
        ("consensus", 0, "crashed: KeyError('boom')", 1),
        ("single", 1, "", 1),
        ("single", 0, "crashed: KeyError('boom')", 1),
    ]
    db.close()


def test_crash_is_recorded_in_the_database_and_raised(monkeypatch, tmp_path):
    def broken_tool(name, tool_input, research):
        raise KeyError("boom")

    monkeypatch.setattr(loop, "run_tool", broken_tool)
    db = Database(tmp_path / "pain.db")

    with pytest.raises(KeyError, match="boom"):
        loop.run_research("niche", StubClient([search_response("q")]), db=db)

    assert rows(db, "SELECT finished, stopped, errors FROM runs") == [(0, "crashed: KeyError('boom')", 1)]
    assert rows(db, "SELECT stage, error FROM errors") == [("crash", "KeyError('boom')")]
    db.close()


def test_failing_source_is_saved_as_an_error(monkeypatch, tmp_path):
    def down(source, query):
        raise ValueError("The forum may be failing")

    monkeypatch.setattr(sources, "search", down)
    db = Database(tmp_path / "pain.db")

    loop.run_research("niche", StubClient([search_response("q")]), max_steps=1, db=db)

    assert rows(db, "SELECT stage, error FROM errors") == [
        ("search", "ValueError: The forum may be failing"),
        ("budget", "step limit of 1 reached before finish"),
    ]
    assert rows(db, "SELECT errors FROM runs") == [(2,)]
    assert rows(db, "SELECT COUNT(*) FROM rejections") == [(0,)]
    db.close()


def test_api_error_and_failed_theme_are_saved_as_errors(stub_search, monkeypatch, tmp_path):
    evidence = [{"source": "n8n", "title": "T", "url": "https://x/1", "quote": "bot goes silent", "date": ""}]
    monkeypatch.setattr(loop.Research, "strong_pains", lambda self: {"Bot goes silent": evidence})

    def failing_theme(niche, pains, client):
        raise loop.ThemeError("cut off")

    monkeypatch.setattr(loop, "find_theme", failing_theme)

    def create(**kwargs):
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))

    db = Database(tmp_path / "pain.db")

    loop.run_research("niche", SimpleNamespace(messages=SimpleNamespace(create=create)), db=db)

    assert [stage for stage, in rows(db, "SELECT stage FROM errors ORDER BY id")] == ["model", "theme"]
    assert rows(db, "SELECT errors FROM runs") == [(2,)]
    db.close()


def test_saved_trace_ends_with_the_finish_result(stub_search, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "min_pains", 0)
    trace_file = tmp_path / "trace.json"

    report("niche", StubClient([tool_use_response("finish", {"note": ""})]), trace_file=trace_file)

    last = json.loads(trace_file.read_text())[-1]
    assert last["role"] == "user"
    assert last["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": '"Finished."', "is_error": False}
    ]


def rows(db, sql):
    with db.engine.connect() as connection:
        return [tuple(row) for row in connection.execute(text(sql))]

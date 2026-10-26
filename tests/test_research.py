from datetime import UTC, datetime, timedelta

import httpx
import pytest

from pain_point_agent import sources
from pain_point_agent.config import settings
from pain_point_agent.judge import Fit, Verdict
from pain_point_agent.research import Research, ResearchError
from pain_point_agent.sources.models import SourceError

QUOTE = "every few days the bot just goes silent"


def thread(topic_id, created_at="2026-10-01T10:00:00Z"):
    return {
        "id": topic_id,
        "title": f"Thread {topic_id}",
        "url": f"https://community.n8n.io/t/x/{topic_id}",
        "posts": [
            {
                "number": 1,
                "author": "a",
                "text": f"Hi.\nEvery few days the bot\njust goes silent ({topic_id}).",
                "created_at": created_at,
            }
        ],
    }


def days_ago(days):
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


class FakeJudge:
    def __init__(self):
        self.calls = []
        self.verdict = Verdict(True, "Describes a bot failing.")

    def __call__(self, quote, post):
        self.calls.append((quote, post))
        return self.verdict


class FakeFit:
    def __init__(self):
        self.calls = []
        self.verdict = Fit(True, "Same problem.")

    def __call__(self, pain, quote, post):
        self.calls.append((pain, quote, post))
        return self.verdict


@pytest.fixture
def judge():
    return FakeJudge()


@pytest.fixture
def fit():
    return FakeFit()


@pytest.fixture
def research(monkeypatch, judge, fit):
    monkeypatch.setattr(sources, "fetch", lambda source, topic_id: thread(topic_id))
    monkeypatch.setattr(sources, "search", lambda source, query: [])
    return Research(judge=judge, fit=fit)


THREE_SOURCES = ["n8n", "openai", "homeassistant"]


def use_all_searches(research):
    for source in THREE_SOURCES:
        for _ in range(settings.max_searches_per_source):
            research.search(source, "q")


def test_record_pain_rejects_thread_that_was_not_fetched(research):
    with pytest.raises(ResearchError, match="not fetched"):
        research.record_pain("Bot goes silent", "n8n", 1, QUOTE)


def test_record_pain_rejects_quote_not_in_thread(research, judge):
    research.fetch("n8n", 1)

    with pytest.raises(ResearchError, match="word for word"):
        research.record_pain("Bot goes silent", "n8n", 1, "the bot crashes every single day")
    assert judge.calls == []


def test_record_pain_rejects_quote_the_judge_says_is_not_a_complaint(research, judge):
    research.fetch("n8n", 1)
    judge.verdict = Verdict(False, "Advice to another user.")

    with pytest.raises(ResearchError, match="not a complaint: Advice to another user."):
        research.record_pain("Bot goes silent", "n8n", 1, QUOTE)
    assert research.pains == {}


def test_judge_sees_the_post_the_quote_came_from(research, judge):
    research.fetch("n8n", 1)
    research.record_pain("Bot goes silent", "n8n", 1, QUOTE)

    assert judge.calls == [(QUOTE, thread(1)["posts"][0]["text"])]


def test_record_pain_rejects_quote_that_does_not_fit_the_problem(research, fit):
    research.fetch("n8n", 1)
    fit.verdict = Fit(False, "The quote is about pricing, not the bot going silent.")

    with pytest.raises(ResearchError, match="does not fit the problem 'Bot goes silent': The quote is about pricing"):
        research.record_pain("Bot goes silent", "n8n", 1, QUOTE)
    assert research.pains == {}


def test_fit_is_not_asked_when_the_quote_is_not_a_complaint(research, judge, fit):
    research.fetch("n8n", 1)
    judge.verdict = Verdict(False, "Advice.")

    with pytest.raises(ResearchError):
        research.record_pain("Bot goes silent", "n8n", 1, QUOTE)
    assert fit.calls == []


def test_fit_sees_the_problem_quote_and_post(research, fit):
    research.fetch("n8n", 1)
    research.record_pain("Bot goes silent", "n8n", 1, QUOTE)

    post = f"Thread: {thread(1)['url']}\n\n{thread(1)['posts'][0]['text']}"
    assert fit.calls == [("Bot goes silent", QUOTE, post)]


def test_record_pain_accepts_quote_with_different_case_and_whitespace(research):
    research.fetch("n8n", 1)

    result = research.record_pain("Bot goes silent", "n8n", 1, "EVERY few days  the bot just goes silent")

    assert result == {"pain": "Bot goes silent", "threads": 1}


def test_record_pain_rejects_same_thread_twice_for_one_pain(research):
    research.fetch("n8n", 1)
    research.record_pain("Bot goes silent", "n8n", 1, QUOTE)

    with pytest.raises(ResearchError, match="already recorded.*counts threads, not quotes"):
        research.record_pain("Bot goes silent", "n8n", 1, QUOTE)


def test_record_pain_rejects_short_quote(research):
    research.fetch("n8n", 1)

    with pytest.raises(ResearchError, match="too short"):
        research.record_pain("Bot goes silent", "n8n", 1, "goes silent")


def test_record_pain_counts_quote_length_after_whitespace_is_collapsed(research):
    research.fetch("n8n", 1)

    with pytest.raises(ResearchError, match="too short"):
        research.record_pain("Bot goes silent", "n8n", 1, "goes          silent        ")


def test_search_limit_is_enforced(research, monkeypatch):
    monkeypatch.setattr(settings, "max_searches", 6)
    monkeypatch.setattr(settings, "max_searches_per_source", 2)
    use_all_searches(research)

    with pytest.raises(ResearchError, match="Search limit"):
        research.search("cursor", "q")


def test_search_limit_per_source_sends_the_agent_elsewhere(research, monkeypatch):
    monkeypatch.setattr(settings, "max_searches_per_source", 2)
    research.search("n8n", "q")
    research.search("n8n", "q")

    with pytest.raises(ResearchError, match="Already 2 searches in n8n. Search a different source"):
        research.search("n8n", "q")
    assert research.searches == 2
    assert research.search("openai", "q") == []


def test_finish_rejected_while_too_few_pains_and_searches_left(research):
    research.fetch("n8n", 1)
    research.record_pain("Bot goes silent", "n8n", 1, QUOTE)

    with pytest.raises(ResearchError, match="Not done"):
        research.finish("")


def test_finish_accepted_when_searches_run_out(research, monkeypatch):
    monkeypatch.setattr(settings, "max_searches", 6)
    monkeypatch.setattr(settings, "max_searches_per_source", 2)
    use_all_searches(research)

    assert research.finish("only noise") == "Finished."
    assert research.finished is True


def test_report_lists_only_pains_with_enough_threads_as_findings(research, monkeypatch):
    monkeypatch.setattr(settings, "min_pains", 1)
    for topic_id in (1, 2, 3):
        research.fetch("n8n", topic_id)
    research.record_pain("Bot goes silent", "n8n", 1, QUOTE)
    research.record_pain("Bot goes silent", "n8n", 2, QUOTE)
    research.record_pain("Webhook drops messages", "n8n", 3, QUOTE)
    research.finish("")

    report = research.report("WhatsApp bots")

    assert "## 1. Bot goes silent\nThreads: 2" in report
    assert "> Thread 1 (2026-10-01): https://community.n8n.io/t/x/1" in report
    assert "## 2." not in report
    assert "- Webhook drops messages: https://community.n8n.io/t/x/3" in report


def test_report_shows_theme_with_what_it_is_based_on(research, monkeypatch):
    monkeypatch.setattr(settings, "min_pains", 1)
    for topic_id in (1, 2):
        research.fetch("n8n", topic_id)
        research.record_pain("Bot goes silent", "n8n", topic_id, QUOTE)
    research.finish("")

    research.theme = "Bots fail without anyone noticing."
    report = research.report("WhatsApp bots")

    assert "## Theme (inference)\nBots fail without anyone noticing.\nBased on: problems 1-1 · 2 threads · sources: n8n" in report
    assert report.index("## Theme") < report.index("## 1. Bot goes silent")


def test_single_thread_pains_do_not_use_up_the_pain_limit(research, monkeypatch):
    monkeypatch.setattr(settings, "max_pains", 1)
    for topic_id in (1, 2, 3):
        research.fetch("n8n", topic_id)
    research.record_pain("Weak one", "n8n", 1, QUOTE)

    research.record_pain("Bot goes silent", "n8n", 2, QUOTE)
    research.record_pain("Bot goes silent", "n8n", 3, QUOTE)

    with pytest.raises(ResearchError, match="Already 1 problems with enough threads"):
        research.record_pain("Weak one", "n8n", 3, QUOTE)


def test_evidence_keeps_the_post_date(research):
    research.fetch("n8n", 1)
    research.record_pain("Bot goes silent", "n8n", 1, QUOTE)

    assert research.pains["Bot goes silent"][0]["date"] == "2026-10-01T10:00:00Z"


def test_old_posts_are_rejected_when_an_age_limit_is_set(research, monkeypatch):
    monkeypatch.setattr(settings, "max_evidence_age_days", 365)
    monkeypatch.setattr(sources, "fetch", lambda source, topic_id: thread(topic_id, days_ago(400 if topic_id == 1 else 10)))
    research.fetch("n8n", 1)
    research.fetch("n8n", 2)

    with pytest.raises(ResearchError, match="older than 365 days"):
        research.record_pain("Bot goes silent", "n8n", 1, QUOTE)
    assert research.record_pain("Bot goes silent", "n8n", 2, QUOTE) == {"pain": "Bot goes silent", "threads": 1}


def test_posts_without_a_date_pass_the_age_limit(research, monkeypatch):
    monkeypatch.setattr(settings, "max_evidence_age_days", 365)
    monkeypatch.setattr(sources, "fetch", lambda source, topic_id: thread(topic_id, ""))
    research.fetch("n8n", 1)

    assert research.record_pain("Bot goes silent", "n8n", 1, QUOTE)["threads"] == 1


def test_same_quote_from_the_same_thread_cannot_support_two_problems(research):
    research.fetch("n8n", 1)
    research.record_pain("Bot goes silent", "n8n", 1, QUOTE)

    with pytest.raises(ResearchError, match="already evidence for 'Bot goes silent'"):
        research.record_pain("No alerts when the bot stops", "n8n", 1, QUOTE)


def test_overlapping_quote_from_the_same_thread_is_caught(research):
    research.fetch("n8n", 1)
    research.record_pain("Bot goes silent", "n8n", 1, QUOTE)

    with pytest.raises(ResearchError, match="already evidence"):
        research.record_pain("No alerts when the bot stops", "n8n", 1, "Every few days the bot")


def test_quote_check_runs_before_the_judges(research, judge, fit):
    research.fetch("n8n", 1)
    research.record_pain("Bot goes silent", "n8n", 1, QUOTE)

    with pytest.raises(ResearchError):
        research.record_pain("No alerts when the bot stops", "n8n", 1, QUOTE)
    assert len(judge.calls) == 1
    assert len(fit.calls) == 1


def test_new_problem_gets_a_list_of_existing_ones(research):
    research.fetch("n8n", 1)
    research.fetch("n8n", 2)
    research.record_pain("Bot goes silent", "n8n", 1, QUOTE)

    result = research.record_pain("Bot stops replying", "n8n", 2, QUOTE)

    assert result["other_problems"] == ["Bot goes silent"]
    assert "same issue" in result["hint"]


def test_adding_to_an_existing_problem_has_no_hint(research):
    research.fetch("n8n", 1)
    research.fetch("n8n", 2)
    research.record_pain("Bot goes silent", "n8n", 1, QUOTE)

    assert research.record_pain("Bot goes silent", "n8n", 2, QUOTE) == {"pain": "Bot goes silent", "threads": 2}


def test_report_of_a_stopped_run_says_so_and_keeps_findings(research, monkeypatch):
    monkeypatch.setattr(settings, "min_pains", 1)
    for topic_id in (1, 2):
        research.fetch("n8n", topic_id)
        research.record_pain("Bot goes silent", "n8n", topic_id, QUOTE)
    research.stopped = "step limit of 20 reached before finish"

    report = research.report("WhatsApp bots")

    assert "Stopped early: step limit of 20 reached before finish." in report
    assert "## 1. Bot goes silent" in report


def test_status_for_model_names_sources_that_are_used_up(research, monkeypatch):
    monkeypatch.setattr(settings, "max_searches", 6)
    monkeypatch.setattr(settings, "max_searches_per_source", 2)
    research.search("n8n", "q")
    research.search("n8n", "q")

    assert research.status_for_model(4) == "Budget: 4 steps left · 4 searches left · no searches left in: n8n"


def test_status_for_model_lists_what_is_already_recorded(research, monkeypatch):
    monkeypatch.setattr(settings, "min_threads_per_pain", 2)
    for topic_id in (1, 2, 3):
        research.fetch("n8n", topic_id)
    research.record_pain("Bot goes silent", "n8n", 1, QUOTE)
    research.record_pain("Bot goes silent", "n8n", 2, QUOTE)
    research.record_pain("Pricing", "n8n", 3, QUOTE)

    assert research.status_for_model(5).endswith(
        "\nRecorded so far, do not record these again: "
        "'Bot goes silent' n8n 1, n8n 2 (enough) · 'Pricing' n8n 3 (needs 1 more thread)"
    )


def test_repeat_thread_rejection_lists_fetched_threads_still_free_for_the_problem(research):
    for topic_id in (1, 2, 3):
        research.fetch("n8n", topic_id)
    research.record_pain("Bot goes silent", "n8n", 1, QUOTE)

    with pytest.raises(ResearchError, match=r"Fetched threads not yet used for this problem: n8n 2, n8n 3\."):
        research.record_pain("Bot goes silent", "n8n", 1, QUOTE)


def test_repeat_thread_rejection_says_to_fetch_more_when_none_are_free(research):
    research.fetch("n8n", 1)
    research.record_pain("Bot goes silent", "n8n", 1, QUOTE)

    with pytest.raises(ResearchError, match="No fetched thread is left for this problem. Fetch another thread"):
        research.record_pain("Bot goes silent", "n8n", 1, QUOTE)


def test_theme_line_counts_threads_not_quotes(research, monkeypatch):
    monkeypatch.setattr(settings, "min_pains", 1)
    monkeypatch.setattr(sources, "fetch", lambda source, topic_id: two_problem_thread(topic_id))
    for topic_id in (1, 2):
        research.fetch("n8n", topic_id)
        research.record_pain("Bot goes silent", "n8n", topic_id, "every few days the bot just goes silent")
        research.record_pain("Pricing is too high", "n8n", topic_id, "the monthly plan is far too expensive")
    research.finish("")

    research.theme = "Bots fail and cost too much."
    report = research.report("WhatsApp bots")

    assert "Based on: problems 1-2 · 4 quotes from 2 threads · sources: n8n" in report


def test_quote_with_line_breaks_stays_on_one_line_in_the_report(research, monkeypatch):
    monkeypatch.setattr(settings, "min_pains", 1)
    for topic_id in (1, 2):
        research.fetch("n8n", topic_id)
        research.record_pain("Bot goes silent", "n8n", topic_id, "Every few days the bot\njust goes silent")
    research.finish("")

    report = research.report("WhatsApp bots")

    assert '> "Every few days the bot just goes silent"' in report


def two_problem_thread(topic_id):
    item = thread(topic_id)
    item["posts"][0]["text"] += " Also the monthly plan is far too expensive."
    return item


def failing_fetch(error):
    def fetch(source, topic_id):
        raise error

    return fetch


def test_source_is_off_after_failing_twice_in_a_row(research, monkeypatch):
    monkeypatch.setattr(settings, "max_source_failures", 2)
    monkeypatch.setattr(sources, "fetch", failing_fetch(SourceError("feed returned nothing")))
    for topic_id in (1, 2):
        with pytest.raises(SourceError):
            research.fetch("appstore", topic_id)

    with pytest.raises(ResearchError, match="appstore failed 2 times in a row and is off for this run"):
        research.fetch("appstore", 3)
    with pytest.raises(ResearchError, match="off for this run"):
        research.search("appstore", "q")
    assert research.searches == 0
    monkeypatch.setattr(sources, "fetch", lambda source, topic_id: thread(topic_id))
    assert research.fetch("n8n", 4)["id"] == 4


def test_http_errors_count_as_source_failures(research, monkeypatch):
    monkeypatch.setattr(settings, "max_source_failures", 1)
    request = httpx.Request("GET", "https://example.test")
    error = httpx.HTTPStatusError("503", request=request, response=httpx.Response(503, request=request))
    def search(source, query):
        raise error

    monkeypatch.setattr(sources, "search", search)
    with pytest.raises(httpx.HTTPStatusError):
        research.search("hn", "q")

    with pytest.raises(ResearchError, match="hn failed 1 times in a row"):
        research.search("hn", "q")


def test_a_success_resets_the_failure_count(research, monkeypatch):
    monkeypatch.setattr(settings, "max_source_failures", 2)
    monkeypatch.setattr(sources, "fetch", failing_fetch(SourceError("down")))
    with pytest.raises(SourceError):
        research.fetch("appstore", 1)
    monkeypatch.setattr(sources, "fetch", lambda source, topic_id: thread(topic_id))
    research.fetch("appstore", 2)
    monkeypatch.setattr(sources, "fetch", failing_fetch(SourceError("down")))

    with pytest.raises(SourceError):
        research.fetch("appstore", 3)
    assert research.failures_in_a_row["appstore"] == 1


def test_an_unusable_item_is_not_a_source_failure(research, monkeypatch):
    monkeypatch.setattr(sources, "fetch", failing_fetch(ValueError("No reviews of 2 stars or less among the latest 50 reviews")))
    for topic_id in (1, 2, 3):
        with pytest.raises(ValueError):
            research.fetch("appstore", topic_id)

    assert research.failures_in_a_row == {}


def test_status_for_model_names_sources_that_are_off(research, monkeypatch):
    monkeypatch.setattr(settings, "max_source_failures", 1)
    monkeypatch.setattr(sources, "fetch", failing_fetch(SourceError("down")))
    with pytest.raises(SourceError):
        research.fetch("appstore", 1)

    assert research.status_for_model(5).endswith("off after failing: appstore")

import json
from pathlib import Path

import pytest

from pain_point_agent.config import settings
from pain_point_agent.sources import discourse
from pain_point_agent.sources.discourse import parse_search, parse_thread
from pain_point_agent.sources.models import SourceError

N8N = "https://community.n8n.io"
FIXTURE = Path(__file__).parent / "fixtures" / "discourse_search.json"


def test_parse_search_returns_trimmed_results():
    results = parse_search(json.loads(FIXTURE.read_text()), N8N)

    assert len(results) == 6
    first = results[0]
    assert set(first) == {"id", "title", "posts_count", "created_at", "url", "blurb"}
    assert first["id"] == 69301
    assert first["posts_count"] == 4
    assert first["url"] == "https://community.n8n.io/t/creating-a-simple-chat-bot-using-whatsapp/69301"
    assert first["blurb"].startswith("Seems simple enough")


def test_parse_search_handles_empty_response():
    assert parse_search({}, N8N) == []


TOPIC_FIXTURE = Path(__file__).parent / "fixtures" / "discourse_topic.json"


def test_parse_thread_returns_plain_text_posts():
    thread = parse_thread(json.loads(TOPIC_FIXTURE.read_text()), N8N)

    assert thread["id"] == 206218
    assert thread["url"] == "https://community.n8n.io/t/whatsapp-bot-responses-not-appearing-in-meta-business-suite-inbox/206218"
    assert [post["number"] for post in thread["posts"]] == [1, 2, 3]
    assert thread["posts"][1]["author"] == "FaithX1"
    assert thread["posts"][1]["created_at"] == "2025-10-15T13:08:14.019Z"
    assert "<" not in thread["posts"][1]["text"]
    assert "do not show in Meta Business Suite Inbox" in thread["posts"][1]["text"]
    assert thread["truncated"] is False


def test_parse_thread_truncates_long_posts(monkeypatch):
    monkeypatch.setattr(settings, "max_post_chars", 50)

    thread = parse_thread(json.loads(TOPIC_FIXTURE.read_text()), N8N)

    assert len(thread["posts"][0]["text"]) == 50
    assert thread["truncated"] is True


def test_fetch_topic_without_posts_says_so(monkeypatch):
    topic = json.loads(TOPIC_FIXTURE.read_text())
    topic["post_stream"]["posts"] = []
    monkeypatch.setattr(discourse, "get", lambda url, params=None: topic)

    with pytest.raises(SourceError, match="came back with no posts"):
        discourse.fetch(N8N, 206218)

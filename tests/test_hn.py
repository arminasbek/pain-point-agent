import json
from pathlib import Path

import pytest

from pain_point_agent import sources
from pain_point_agent.config import settings
from pain_point_agent.sources import hn
from pain_point_agent.sources.hn import parse_search, parse_thread

API = "https://hn.algolia.com/api/v1"
COMMENTS_FIXTURE = Path(__file__).parent / "fixtures" / "hn_comments.json"
STORIES_FIXTURE = Path(__file__).parent / "fixtures" / "hn_stories.json"
ITEM_FIXTURE = Path(__file__).parent / "fixtures" / "hn_item.json"
COMMENT_FIXTURE = Path(__file__).parent / "fixtures" / "hn_comment_item.json"
TITLE = "I tried to build a WhatsApp bot. Meta banned me before it left the drawing board"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_parse_search_returns_matched_comment_per_story():
    results = parse_search(load(COMMENTS_FIXTURE), load(STORIES_FIXTURE))

    assert len(results) == 10
    first = results[0]
    assert set(first) == {"id", "title", "posts_count", "created_at", "url", "blurb"}
    assert first["id"] == 46922154
    assert first["title"] == "Facebook seemingly randomly bans tons of users"
    assert first["posts_count"] == 3
    assert first["created_at"] == "2026-02-07T08:00:08Z"
    assert first["url"] == "https://news.ycombinator.com/item?id=46922153"
    assert first["blurb"].startswith("This is a subreddit full of users")


def test_parse_search_keeps_first_comment_per_story():
    comments = load(COMMENTS_FIXTURE)
    duplicate = dict(comments["hits"][0], objectID="1", comment_text="a later comment in the same story")
    comments["hits"].insert(1, duplicate)

    results = parse_search(comments, load(STORIES_FIXTURE))

    assert len(results) == 10
    assert results[0]["blurb"].startswith("This is a subreddit full of users")


def test_parse_search_unknown_story_count_is_zero():
    assert parse_search(load(COMMENTS_FIXTURE), {})[0]["posts_count"] == 0


def test_parse_search_handles_empty_response():
    assert parse_search({}, {}) == []


def test_parse_thread_flattens_comment_tree_depth_first():
    thread = parse_thread(load(ITEM_FIXTURE), TITLE)

    assert thread["id"] == 46060177
    assert thread["title"] == TITLE
    assert thread["url"] == "https://news.ycombinator.com/item?id=46060177"
    assert [post["author"] for post in thread["posts"]] == ["jkhall81", "arresin", "ronsor", "AlgebraFox"]
    assert [post["number"] for post in thread["posts"]] == [1, 2, 3, 4]
    assert thread["posts"][0]["created_at"] == "2025-11-26T17:51:06.000Z"
    assert thread["posts_count"] == 4
    assert "banned totally for no reason" in thread["posts"][3]["text"]
    assert thread["truncated"] is False


def test_parse_thread_skips_deleted_comment_but_keeps_its_replies():
    data = load(ITEM_FIXTURE)
    reply = {"id": 2, "type": "comment", "author": "someone", "text": "same thing happened to me", "children": []}
    data["children"].append({"id": 1, "type": "comment", "author": None, "text": None, "children": [reply]})

    thread = parse_thread(data, TITLE)

    assert [post["author"] for post in thread["posts"]][-1] == "someone"
    assert thread["posts_count"] == 5


def test_parse_thread_converts_html_to_plain_text():
    data = load(ITEM_FIXTURE)
    data["children"] = [
        {"id": 3, "type": "comment", "author": "a", "text": "<p>it&#x27;s broken<p>second line", "children": []}
    ]

    text = parse_thread(data, TITLE)["posts"][0]["text"]

    assert "<" not in text
    assert "it's broken" in text


def test_parse_thread_keeps_story_text_as_first_post():
    data = load(ITEM_FIXTURE)
    data["text"] = "Ask HN: my bot keeps getting banned"

    thread = parse_thread(data, TITLE)

    assert thread["posts"][0]["author"] == "rpgbr"
    assert thread["posts"][0]["text"] == "Ask HN: my bot keeps getting banned"


def test_parse_thread_of_comment_starts_with_it_and_links_its_story():
    thread = parse_thread(load(COMMENT_FIXTURE), TITLE)

    assert thread["id"] == 46060274
    assert thread["url"] == "https://news.ycombinator.com/item?id=46060177"
    assert [post["author"] for post in thread["posts"]] == ["jkhall81", "arresin"]
    assert thread["posts"][0]["text"] == "I hate Meta and all their products."


def test_parse_thread_truncates_long_posts(monkeypatch):
    monkeypatch.setattr(settings, "max_post_chars", 20)

    thread = parse_thread(load(ITEM_FIXTURE), TITLE)

    assert all(len(post["text"]) <= 20 for post in thread["posts"])
    assert thread["truncated"] is True


def fake_get(monkeypatch, *responses: dict) -> list:
    calls = []
    replies = iter(responses)

    def get(url, params=None):
        calls.append((url, params))
        return next(replies)

    monkeypatch.setattr(hn, "get", get)
    return calls


def test_search_finds_comments_then_counts_their_stories(monkeypatch):
    calls = fake_get(monkeypatch, load(COMMENTS_FIXTURE), load(STORIES_FIXTURE))

    results = hn.search(API, "whatsapp bot banned")

    assert calls[0] == (
        f"{API}/search",
        {"query": "whatsapp bot banned", "tags": "comment", "hitsPerPage": str(settings.search_limit)},
    )
    url, params = calls[1]
    assert url == f"{API}/search"
    assert params["tags"].startswith("story,(story_46922153,story_14322607,")
    assert params["hitsPerPage"] == "10"
    assert len(results) == 10


def test_search_without_comments_makes_one_call(monkeypatch):
    calls = fake_get(monkeypatch, {"hits": []})

    assert hn.search(API, "nothing matches this") == []
    assert len(calls) == 1


def test_fetch_story_uses_its_own_title(monkeypatch):
    calls = fake_get(monkeypatch, load(ITEM_FIXTURE))

    thread = hn.fetch(API, 46060177)

    assert calls == [(f"{API}/items/46060177", None)]
    assert thread["title"] == TITLE


def test_fetch_comment_looks_up_story_title(monkeypatch):
    calls = fake_get(monkeypatch, load(COMMENT_FIXTURE), {"hits": [{"title": TITLE}]})

    thread = hn.fetch(API, 46060274)

    assert calls == [
        (f"{API}/items/46060274", None),
        (f"{API}/search", {"tags": "story,story_46060177", "hitsPerPage": "1"}),
    ]
    assert thread["title"] == TITLE


def test_fetch_item_without_text_says_so(monkeypatch):
    fake_get(monkeypatch, {"id": 5, "type": "comment", "author": None, "text": None, "children": []}, {"hits": []})

    with pytest.raises(ValueError, match="Item 5 has no text"):
        hn.fetch(API, 5)


def test_hn_is_a_configured_source():
    assert settings.sources["hn"].type == "hn"
    assert settings.sources["hn"].url == API
    assert sources.ADAPTERS["hn"] is hn

import json
from pathlib import Path

import pytest

from pain_point_agent import sources
from pain_point_agent.config import settings
from pain_point_agent.sources import appstore
from pain_point_agent.sources.appstore import parse_reviews, parse_search, parse_thread
from pain_point_agent.sources.models import SourceError

API = "https://itunes.apple.com"
FIXTURES = Path(__file__).parent / "fixtures"
APP_ID = 1438388363


def load(name: str):
    return json.loads((FIXTURES / name).read_text())


def test_parse_search_puts_most_rated_apps_first():
    results = parse_search(load("appstore_search.json"))

    assert len(results) == 23
    first = results[0]
    assert set(first) == {"id", "title", "posts_count", "created_at", "url", "blurb"}
    assert first["id"] == 1528595748
    assert first["title"] == "Finch: Self-Care Pet"
    assert first["posts_count"] == 765007
    assert first["created_at"] == "2021-05-12T07:00:00Z"
    assert first["url"] == "https://apps.apple.com/us/app/finch-self-care-pet/id1528595748?uo=4"
    assert first["blurb"].startswith("Meet your new self-care best friend!")
    assert [result["posts_count"] for result in results] == sorted(
        (result["posts_count"] for result in results), reverse=True
    )


def test_parse_search_handles_empty_response():
    assert parse_search({}) == []


def test_parse_reviews_keeps_only_one_and_two_star_reviews():
    reviews = parse_reviews(load("appstore_reviews.json"))

    assert len(reviews) == 9
    assert reviews[0] == (
        "Hdjnendndksm",
        "Paying\nYou should not have to pay Tudor something for your mental health",
        "2026-10-13T14:05:07-07:00",
    )


def test_parse_reviews_max_rating_comes_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "appstore_max_rating", 3)

    assert len(parse_reviews(load("appstore_reviews.json"))) == 15


def test_parse_reviews_handles_a_single_review_object():
    entry = load("appstore_reviews.json")["feed"]["entry"][0]
    entry["im:rating"]["label"] = "1"

    assert len(parse_reviews({"feed": {"entry": entry}})) == 1


def test_parse_reviews_handles_a_page_without_reviews():
    assert parse_reviews({"feed": {}}) == []


def test_parse_thread_turns_app_reviews_into_posts():
    app = load("appstore_lookup.json")["results"][0]
    thread = parse_thread(app, parse_reviews(load("appstore_reviews.json")))

    assert thread["id"] == APP_ID
    assert thread["title"] == "Habit Tracker"
    assert thread["url"] == "https://apps.apple.com/us/app/habit-tracker/id1438388363?uo=4"
    assert thread["posts_count"] == 9
    assert [post["number"] for post in thread["posts"]] == list(range(1, 10))
    assert thread["posts"][0]["author"] == "Hdjnendndksm"
    assert thread["posts"][0]["created_at"] == "2026-10-13T14:05:07-07:00"
    assert thread["truncated"] is False


def fake_get(monkeypatch, *responses) -> list:
    calls = []
    replies = iter(responses)

    def get(url, params=None, headers=None):
        calls.append((url, params))
        return next(replies)

    monkeypatch.setattr(appstore, "get", get)
    return calls


def test_search_asks_for_apps_in_the_configured_country(monkeypatch):
    calls = fake_get(monkeypatch, load("appstore_search.json"))

    results = appstore.search(API, "habit tracker")

    assert calls == [
        (
            f"{API}/search",
            {
                "term": "habit tracker",
                "entity": "software",
                "limit": str(settings.appstore_search_pool),
                "country": settings.appstore_country,
            },
        )
    ]
    assert len(results) == settings.search_limit


def reviews_url(page: int) -> str:
    return f"{API}/{settings.appstore_country}/rss/customerreviews/page={page}/sortBy=mostRecent/id={APP_ID}/json"


def test_fetch_reads_review_pages_until_one_is_empty(monkeypatch):
    calls = fake_get(monkeypatch, load("appstore_lookup.json"), load("appstore_reviews.json"), {"feed": {}})

    thread = appstore.fetch(API, APP_ID)

    assert calls == [
        (f"{API}/lookup", {"id": str(APP_ID), "country": settings.appstore_country}),
        (reviews_url(1), None),
        (reviews_url(2), None),
    ]
    assert thread["posts_count"] == 9


def test_fetch_stops_after_the_configured_number_of_pages(monkeypatch):
    monkeypatch.setattr(settings, "appstore_review_pages", 2)
    page = load("appstore_reviews.json")
    calls = fake_get(monkeypatch, load("appstore_lookup.json"), page, page)

    thread = appstore.fetch(API, APP_ID)

    assert len(calls) == 3
    assert thread["posts_count"] == 18


def test_fetch_empty_review_feed_says_the_feed_may_be_down(monkeypatch):
    fake_get(monkeypatch, load("appstore_lookup.json"), {"feed": {}})

    with pytest.raises(SourceError, match="review feed returned nothing"):
        appstore.fetch(API, APP_ID)


def test_fetch_without_low_ratings_says_how_many_reviews_were_read(monkeypatch):
    page = load("appstore_reviews.json")
    for entry in page["feed"]["entry"]:
        entry["im:rating"]["label"] = "5"
    fake_get(monkeypatch, load("appstore_lookup.json"), page, {"feed": {}})

    with pytest.raises(ValueError, match=f"No reviews of 2 stars or less among the latest {len(page['feed']['entry'])} reviews"):
        appstore.fetch(API, APP_ID)


def test_fetch_unknown_app_says_so(monkeypatch):
    fake_get(monkeypatch, {"resultCount": 0, "results": []})

    with pytest.raises(ValueError, match="No App Store app"):
        appstore.fetch(API, 1)


def test_appstore_is_a_configured_source():
    assert settings.sources["appstore"].type == "appstore"
    assert settings.sources["appstore"].url == API
    assert sources.ADAPTERS["appstore"] is appstore

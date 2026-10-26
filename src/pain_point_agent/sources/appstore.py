# Disabled since 2026-10-19: Apple reviews RSS returns no reviews even for apps with thousands of ratings. Re-enable with enabled=True in config.py.
from typing import Any

from pain_point_agent.config import settings
from pain_point_agent.sources.models import SearchResult, SourceError, Thread
from pain_point_agent.sources.utils.http_client import get
from pain_point_agent.sources.utils.posts import trim_posts


def parse_search(data: dict[str, Any]) -> list[SearchResult]:
    apps = sorted(data.get("results", []), key=lambda app: app.get("userRatingCount") or 0, reverse=True)
    return [
        SearchResult(
            id=app["trackId"],
            title=app["trackName"],
            posts_count=app.get("userRatingCount") or 0,
            created_at=app.get("releaseDate", ""),
            url=app["trackViewUrl"],
            blurb=(app.get("description") or "")[: settings.max_blurb_chars],
        )
        for app in apps
    ]


def parse_reviews(data: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [
        (
            entry["author"]["name"]["label"],
            f"{entry['title']['label']}\n{entry['content']['label']}".strip(),
            entry.get("updated", {}).get("label", ""),
        )
        for entry in _entries(data)
        if int(entry["im:rating"]["label"]) <= settings.appstore_max_rating
    ]


def parse_thread(app: dict[str, Any], reviews: list[tuple[str, str, str]]) -> Thread:
    posts, truncated = trim_posts((number, *review) for number, review in enumerate(reviews, start=1))
    return Thread(
        id=app["trackId"],
        title=app["trackName"],
        url=app["trackViewUrl"],
        posts_count=len(reviews),
        posts=posts,
        truncated=truncated,
    )


def search(base_url: str, query: str) -> list[SearchResult]:
    params = {
        "term": query,
        "entity": "software",
        "limit": str(settings.appstore_search_pool),
        "country": settings.appstore_country,
    }
    return parse_search(get(f"{base_url}/search", params))[: settings.search_limit]


def fetch(base_url: str, item_id: int) -> Thread:
    country = settings.appstore_country
    apps = get(f"{base_url}/lookup", {"id": str(item_id), "country": country}).get("results", [])
    if not apps:
        raise ValueError(f"No App Store app with id {item_id}.")
    reviews: list[tuple[str, str, str]] = []
    seen = 0
    for page in range(1, settings.appstore_review_pages + 1):
        data = get(f"{base_url}/{country}/rss/customerreviews/page={page}/sortBy=mostRecent/id={item_id}/json")
        entries = _entries(data)
        if not entries:
            break
        seen += len(entries)
        reviews += parse_reviews(data)
    if not seen:
        raise SourceError(f"The review feed returned nothing for app {item_id}. It may be down, use another source.")
    if not reviews:
        raise ValueError(
            f"No reviews of {settings.appstore_max_rating} stars or less among the latest {seen} reviews of app {item_id}. "
            "Fetch another app."
        )
    return parse_thread(apps[0], reviews)


def _entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    entry = data.get("feed", {}).get("entry", [])
    return [entry] if isinstance(entry, dict) else entry

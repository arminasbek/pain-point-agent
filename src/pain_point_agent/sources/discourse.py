from typing import Any

from pain_point_agent.config import settings
from pain_point_agent.sources.utils.http_client import get
from pain_point_agent.sources.utils.posts import trim_posts
from pain_point_agent.sources.models import SearchResult, SourceError, Thread
from pain_point_agent.sources.utils.text import html_to_text


def parse_search(data: dict[str, Any], base_url: str) -> list[SearchResult]:
    blurbs = {post["topic_id"]: post["blurb"] for post in reversed(data.get("posts", []))}
    return [
        SearchResult(
            id=topic["id"],
            title=topic["title"],
            posts_count=topic["posts_count"],
            created_at=topic["created_at"],
            url=_thread_url(base_url, topic["slug"], topic["id"]),
            blurb=blurbs.get(topic["id"], ""),
        )
        for topic in data.get("topics", [])
    ]


def parse_thread(data: dict[str, Any], base_url: str) -> Thread:
    posts, truncated = trim_posts(
        (raw["post_number"], raw["username"], html_to_text(raw["cooked"]), raw.get("created_at", ""))
        for raw in data["post_stream"]["posts"]
    )
    return Thread(
        id=data["id"],
        title=data["title"],
        url=_thread_url(base_url, data["slug"], data["id"]),
        posts_count=data["posts_count"],
        posts=posts,
        truncated=truncated or len(data["post_stream"]["posts"]) < data["posts_count"],
    )


def search(base_url: str, query: str) -> list[SearchResult]:
    data = get(f"{base_url}/search.json", {"q": query})
    return parse_search(data, base_url)[: settings.search_limit]


def fetch(base_url: str, topic_id: int) -> Thread:
    thread = parse_thread(get(f"{base_url}/t/{topic_id}.json"), base_url)
    if not thread["posts"]:
        raise SourceError(f"Topic {topic_id} came back with no posts after parsing and trimming. Fetch another topic or use another source.")
    return thread


def _thread_url(base_url: str, slug: str, topic_id: int) -> str:
    return f"{base_url}/t/{slug}/{topic_id}"

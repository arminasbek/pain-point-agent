from collections.abc import Iterator
from typing import Any

from pain_point_agent.config import settings
from pain_point_agent.sources.utils.http_client import get
from pain_point_agent.sources.utils.posts import trim_posts
from pain_point_agent.sources.models import SearchResult, Thread
from pain_point_agent.sources.utils.text import html_to_text


def parse_search(comments: dict[str, Any], stories: dict[str, Any]) -> list[SearchResult]:
    counts = {int(hit["objectID"]): hit.get("num_comments") or 0 for hit in stories.get("hits", [])}
    results: dict[int, SearchResult] = {}
    for hit in comments.get("hits", []):
        story_id = hit["story_id"]
        if story_id in results:
            continue
        results[story_id] = SearchResult(
            id=int(hit["objectID"]),
            title=hit.get("story_title") or "",
            posts_count=counts.get(story_id, 0),
            created_at=hit["created_at"],
            url=_item_url(story_id),
            blurb=html_to_text(hit.get("comment_text") or "")[: settings.max_blurb_chars],
        )
    return list(results.values())


def parse_thread(data: dict[str, Any], title: str) -> Thread:
    items = [item for item in _walk(data) if item.get("text")]
    posts, truncated = trim_posts(
        (number, item["author"], html_to_text(item["text"]), item.get("created_at", ""))
        for number, item in enumerate(items, start=1)
    )
    return Thread(
        id=data["id"],
        title=title,
        url=_item_url(data.get("story_id") or data["id"]),
        posts_count=len(items),
        posts=posts,
        truncated=truncated,
    )


def search(base_url: str, query: str) -> list[SearchResult]:
    comments = get(f"{base_url}/search", {"query": query, "tags": "comment", "hitsPerPage": str(settings.search_limit)})
    story_ids = list(dict.fromkeys(hit["story_id"] for hit in comments.get("hits", [])))
    if not story_ids:
        return []
    tags = "story,(" + ",".join(f"story_{story_id}" for story_id in story_ids) + ")"
    stories = get(f"{base_url}/search", {"tags": tags, "hitsPerPage": str(len(story_ids))})
    return parse_search(comments, stories)


def fetch(base_url: str, item_id: int) -> Thread:
    item = get(f"{base_url}/items/{item_id}")
    title = item.get("title") or _story_title(base_url, item.get("story_id") or item_id)
    thread = parse_thread(item, title)
    if not thread["posts"]:
        raise ValueError(f"Item {item_id} has no text, it may be deleted. Fetch another result.")
    return thread


def _item_url(item_id: int) -> str:
    return f"https://news.ycombinator.com/item?id={item_id}"


def _walk(item: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield item
    for child in item.get("children", []):
        yield from _walk(child)


def _story_title(base_url: str, story_id: int) -> str:
    stories = get(f"{base_url}/search", {"tags": f"story,story_{story_id}", "hitsPerPage": "1"})
    hits = stories.get("hits", [])
    return hits[0]["title"] if hits else ""

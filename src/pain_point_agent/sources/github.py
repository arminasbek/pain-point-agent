import re
from typing import Any

from pain_point_agent.config import settings
from pain_point_agent.sources.models import SearchResult, Thread
from pain_point_agent.sources.utils.http_client import get
from pain_point_agent.sources.utils.posts import trim_posts

HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

_issue_urls: dict[int, str] = {}


def parse_search(data: dict[str, Any]) -> list[SearchResult]:
    return [
        SearchResult(
            id=item["id"],
            title=item["title"],
            posts_count=item["comments"] + 1,
            created_at=item["created_at"],
            url=item["html_url"],
            blurb=_clean(item.get("body"))[: settings.max_blurb_chars],
        )
        for item in data.get("items", [])
    ]


def parse_thread(issue: dict[str, Any], comments: list[dict[str, Any]]) -> Thread:
    items = [(issue["user"]["login"], _clean(issue.get("body")), issue.get("created_at", ""))]
    items += [
        (comment["user"]["login"], _clean(comment.get("body")), comment.get("created_at", "")) for comment in comments
    ]
    items = [item for item in items if item[1]]
    posts, truncated = trim_posts((number, *item) for number, item in enumerate(items, start=1))
    return Thread(
        id=issue["id"],
        title=issue["title"],
        url=issue["html_url"],
        posts_count=issue["comments"] + 1,
        posts=posts,
        truncated=truncated or len(comments) < issue["comments"],
    )


def search(base_url: str, query: str) -> list[SearchResult]:
    params = {"q": f"{query} is:issue comments:>1", "per_page": str(settings.search_limit)}
    data = get(f"{base_url}/search/issues", params, _headers())
    for item in data.get("items", []):
        _issue_urls[item["id"]] = item["url"]
    return parse_search(data)[: settings.search_limit]


def fetch(base_url: str, item_id: int) -> Thread:
    url = _issue_urls.get(item_id)
    if url is None:
        raise ValueError(f"Issue {item_id} did not come from a GitHub search in this run. Search first.")
    issue = get(url, headers=_headers())
    comments = get(issue["comments_url"], {"per_page": str(settings.github_max_comments)}, _headers())
    thread = parse_thread(issue, comments)
    if not thread["posts"]:
        raise ValueError(f"Issue {item_id} and its comments have no text. Fetch another result.")
    return thread


def _clean(body: str | None) -> str:
    return HTML_COMMENT.sub("", body or "").strip()


def _headers() -> dict[str, str]:
    token = settings.github_token
    return {"Authorization": f"Bearer {token.get_secret_value()}"} if token else {}

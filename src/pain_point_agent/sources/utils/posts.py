from collections.abc import Iterable

from pain_point_agent.config import settings
from pain_point_agent.sources.models import Post


def trim_posts(items: Iterable[tuple[int, str, str, str]]) -> tuple[list[Post], bool]:
    posts: list[Post] = []
    budget = settings.max_thread_chars
    truncated = False
    for number, author, text, created_at in items:
        if len(text) > settings.max_post_chars:
            text, truncated = text[: settings.max_post_chars], True
        if len(text) > budget:
            truncated = True
            break
        budget -= len(text)
        posts.append(Post(number=number, author=author, text=text, created_at=created_at))
    return posts, truncated

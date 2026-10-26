from typing import TypedDict


class SourceError(Exception):
    pass


class SearchResult(TypedDict):
    id: int
    title: str
    posts_count: int
    created_at: str
    url: str
    blurb: str


class Post(TypedDict):
    number: int
    author: str
    text: str
    created_at: str


class Thread(TypedDict):
    id: int
    title: str
    url: str
    posts_count: int
    posts: list[Post]
    truncated: bool

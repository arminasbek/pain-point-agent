import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, TypedDict, TypeVar

import httpx

from pain_point_agent import sources
from pain_point_agent.config import settings
from pain_point_agent.judge import Fit, Verdict, in_thread
from pain_point_agent.sources.models import Post, SearchResult, SourceError, Thread

Result = TypeVar("Result")

QUOTE_CHARS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


class ResearchError(Exception):
    pass


class Evidence(TypedDict):
    source: str
    title: str
    url: str
    quote: str
    date: str


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(QUOTE_CHARS)).strip().casefold()


def overlaps(quote: str, other: str) -> bool:
    a, b = normalize(quote), normalize(other)
    return a in b or b in a


def too_old(created_at: str) -> bool:
    if not created_at or settings.max_evidence_age_days is None:
        return False
    posted = datetime.fromisoformat(created_at)
    if posted.tzinfo is None:
        posted = posted.replace(tzinfo=UTC)
    return datetime.now(UTC) - posted > timedelta(days=settings.max_evidence_age_days)


@dataclass
class Research:
    judge: Callable[[str, str], Verdict]
    fit: Callable[[str, str, str], Fit]
    searches: int = 0
    searches_by_source: dict[str, int] = field(default_factory=dict)
    failures_in_a_row: dict[str, int] = field(default_factory=dict)
    broken_sources: set[str] = field(default_factory=set)
    fetched: dict[tuple[str, int], Thread] = field(default_factory=dict)
    pains: dict[str, list[Evidence]] = field(default_factory=dict)
    finished: bool = False
    note: str = ""
    stopped: str = ""
    theme: str | None = None
    consensus: str = ""
    usage: dict[str, float] = field(default_factory=dict)

    def search(self, source: str, query: str) -> list[SearchResult]:
        self._check_source_available(source)
        if self.searches >= settings.max_searches:
            raise ResearchError(f"Search limit of {settings.max_searches} reached. Fetch, record_pain or finish.")
        used = self.searches_by_source.get(source, 0)
        if used >= settings.max_searches_per_source:
            raise ResearchError(f"Already {used} searches in {source}. Search a different source that fits the niche.")
        self.searches += 1
        self.searches_by_source[source] = used + 1
        return self._call_source(source, sources.search, query)

    def fetch(self, source: str, id: int) -> Thread:
        self._check_source_available(source)
        thread = self._call_source(source, sources.fetch, id)
        self.fetched[(source, id)] = thread
        return thread

    def record_pain(self, pain: str, source: str, id: int, quote: str) -> dict[str, Any]:
        thread, post = self._find_quoted_post(source, id, quote)
        self._check_pain_limit(pain)
        self._check_not_counted_twice(pain, thread, quote)
        self._check_meaningful(pain, quote, thread, post)
        return self._add(pain, source, thread, post, quote)

    def finish(self, note: str) -> str:
        strong = self.strong_pains()
        if len(strong) < settings.min_pains and self.searches < settings.max_searches:
            weak = [pain for pain in self.pains if pain not in strong]
            raise ResearchError(
                f"Not done: {len(strong)} pains with {settings.min_threads_per_pain}+ threads, need "
                f"{settings.min_pains}. Needing more threads: {weak}. "
                f"{settings.max_searches - self.searches} searches left."
            )
        self.finished = True
        self.note = note
        return "Finished."

    def strong_pains(self) -> dict[str, list[Evidence]]:
        return {pain: evidence for pain, evidence in self.pains.items() if len(evidence) >= settings.min_threads_per_pain}

    def status_for_model(self, steps_left: int) -> str:
        parts = [f"Budget: {steps_left} steps left", f"{settings.max_searches - self.searches} searches left"]
        used_up = [name for name, used in self.searches_by_source.items() if used >= settings.max_searches_per_source]
        if used_up:
            parts.append(f"no searches left in: {', '.join(used_up)}")
        if self.broken_sources:
            parts.append(f"off after failing: {', '.join(sorted(self.broken_sources))}")
        return " · ".join(parts) + self._recorded()

    def report(self, niche: str) -> str:
        theme = self.theme
        lines = [f"# Pain report: {niche}", "", f"Searches: {self.searches} · Threads read: {len(self.fetched)}"]
        if self.stopped:
            lines += ["", f"Stopped early: {self.stopped}. Findings below are what was verified before the stop."]
        strong = self.strong_pains()
        if theme:
            all_evidence = [item for items in strong.values() for item in items]
            source_names = ", ".join(sorted({item["source"] for item in all_evidence}))
            threads = len({item["url"] for item in all_evidence})
            counted = f"{threads} threads" if threads == len(all_evidence) else f"{len(all_evidence)} quotes from {threads} threads"
            lines += [
                "",
                "## Theme (inference)",
                theme,
                f"Based on: problems 1-{len(strong)} · {counted} · sources: {source_names}",
            ]
        for number, (pain, evidence) in enumerate(strong.items(), 1):
            lines += ["", f"## {number}. {pain}", f"Threads: {len(evidence)}"]
            for item in evidence:
                date = f" ({item['date'][:10]})" if item["date"] else ""
                quote = " ".join(item["quote"].split())
                lines += ["", f'> "{quote}"', f"> {item['title']}{date}: {item['url']}"]
        weak = {pain: evidence for pain, evidence in self.pains.items() if pain not in strong}
        if weak:
            lines += ["", "## Single-thread leads (not enough evidence)"]
            lines += [f"- {pain}: {evidence[0]['url']}" for pain, evidence in weak.items()]
        if self.consensus:
            lines += ["", self.consensus]
        if self.note:
            lines += ["", f"Agent's note (written by the model, not checked by code): {self.note}"]
        return "\n".join(lines)

    def _check_source_available(self, source: str) -> None:
        if source in self.broken_sources:
            raise ResearchError(
                f"{source} failed {settings.max_source_failures} times in a row and is off for this run. Use another source."
            )

    def _call_source(self, source: str, call: Callable[..., Result], *args: Any) -> Result:
        try:
            result = call(source, *args)
        except (SourceError, httpx.HTTPError):
            self._record_failure(source)
            raise
        self.failures_in_a_row[source] = 0
        return result

    def _record_failure(self, source: str) -> None:
        self.failures_in_a_row[source] = self.failures_in_a_row.get(source, 0) + 1
        if self.failures_in_a_row[source] >= settings.max_source_failures:
            self.broken_sources.add(source)

    def _find_quoted_post(self, source: str, id: int, quote: str) -> tuple[Thread, Post]:
        thread = self.fetched.get((source, id))
        if thread is None:
            raise ResearchError(f"Thread {id} from {source} was not fetched. Fetch it first.")
        if len(normalize(quote)) < settings.min_quote_chars:
            raise ResearchError(f"Quote too short, use at least {settings.min_quote_chars} characters.")
        post = next((post for post in thread["posts"] if normalize(quote) in normalize(post["text"])), None)
        if post is None:
            raise ResearchError(f"Quote not found word for word in thread {id}. Copy it exactly from one post.")
        if too_old(post["created_at"]):
            raise ResearchError(
                f"Post from {post['created_at'][:10]} is older than {settings.max_evidence_age_days} days. Use recent evidence."
            )
        return thread, post

    def _check_pain_limit(self, pain: str) -> None:
        strong = self.strong_pains()
        if pain not in strong and len(strong) >= settings.max_pains:
            raise ResearchError(
                f"Already {settings.max_pains} problems with enough threads. Add to one of {list(strong)} or finish."
            )

    def _check_not_counted_twice(self, pain: str, thread: Thread, quote: str) -> None:
        used = {item["url"] for item in self.pains.get(pain, [])}
        if thread["url"] in used:
            free = [f"{name} {key}" for (name, key), item in self.fetched.items() if item["url"] not in used]
            hint = (
                f"Fetched threads not yet used for this problem: {', '.join(free)}."
                if free
                else "No fetched thread is left for this problem. Fetch another thread or move on."
            )
            raise ResearchError(
                "This thread is already recorded for this pain. A problem counts threads, not quotes: "
                f"more quotes from the same thread add nothing. Record a different thread, or move on. {hint}"
            )
        for other, items in self.pains.items():
            for item in items:
                if other != pain and item["url"] == thread["url"] and overlaps(quote, item["quote"]):
                    raise ResearchError(
                        f"This quote is already evidence for '{other}'. One quote supports one problem, "
                        "keep it under the problem it describes best."
                    )

    def _check_meaningful(self, pain: str, quote: str, thread: Thread, post: Post) -> None:
        verdict = self.judge(quote, post["text"])
        if not verdict.complaint:
            raise ResearchError(
                f"Quote is not a complaint: {verdict.reason} Use a quote where someone describes a problem they have."
            )
        fit = self.fit(pain, quote, in_thread(thread["url"], post["text"]))
        if not fit.fits:
            raise ResearchError(
                f"Quote does not fit the problem '{pain}': {fit.reason} Record it under a problem it describes, or use another quote."
            )

    def _add(self, pain: str, source: str, thread: Thread, post: Post, quote: str) -> dict[str, Any]:
        new = pain not in self.pains
        evidence = self.pains.setdefault(pain, [])
        evidence.append(
            Evidence(source=source, title=thread["title"], url=thread["url"], quote=quote, date=post["created_at"])
        )
        result: dict[str, Any] = {"pain": pain, "threads": len(evidence)}
        others = [name for name in self.pains if name != pain]
        if new and others:
            result["other_problems"] = others
            result["hint"] = "If this is the same issue as one of other_problems, add evidence under that exact name instead."
        return result

    def _recorded(self) -> str:
        if not self.pains:
            return ""
        names = {thread["url"]: f"{source} {id}" for (source, id), thread in self.fetched.items()}
        problems = []
        for pain, items in self.pains.items():
            missing = settings.min_threads_per_pain - len(items)
            status = "enough" if missing <= 0 else f"needs {missing} more thread{'s' if missing > 1 else ''}"
            problems.append(f"'{pain}' {', '.join(names[item['url']] for item in items)} ({status})")
        return f"\nRecorded so far, do not record these again: {' · '.join(problems)}"

import json
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import ForeignKey, Text, UniqueConstraint, create_engine, event, func, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from pain_point_agent.config import settings
from pain_point_agent.research import Research

class Base(DeclarativeBase):
    pass

class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"))
    kind: Mapped[str]
    niche: Mapped[str]
    model: Mapped[str]
    started_at: Mapped[str]
    finished_at: Mapped[str | None]
    finished: Mapped[bool | None]
    stopped: Mapped[str | None] = mapped_column(Text)
    theme: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)
    consensus: Mapped[str | None] = mapped_column(Text)
    searches: Mapped[int | None]
    errors: Mapped[int | None]
    cost_usd: Mapped[float | None]
    usage: Mapped[str | None] = mapped_column(Text)
    report: Mapped[str | None] = mapped_column(Text)

class Search(Base):
    __tablename__ = "searches"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    source: Mapped[str]
    query: Mapped[str] = mapped_column(Text)
    results: Mapped[int]

class Thread(Base):
    __tablename__ = "threads"
    __table_args__ = (UniqueConstraint("source", "item_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str]
    item_id: Mapped[int]
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    posts_count: Mapped[int | None]
    last_fetched_at: Mapped[str]

class Post(Base):
    __tablename__ = "posts"

    thread_id: Mapped[int] = mapped_column(ForeignKey("threads.id"), primary_key=True)
    number: Mapped[int] = mapped_column(primary_key=True)
    author: Mapped[str | None]
    created_at: Mapped[str | None]
    text: Mapped[str] = mapped_column(Text)

class RunThread(Base):
    __tablename__ = "run_threads"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    thread_id: Mapped[int] = mapped_column(ForeignKey("threads.id"), primary_key=True)

class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    pain: Mapped[str] = mapped_column(Text)
    strong: Mapped[bool]
    source: Mapped[str]
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    quote: Mapped[str] = mapped_column(Text)
    post_date: Mapped[str | None]

class Rejection(Base):
    __tablename__ = "rejections"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    tool: Mapped[str]
    input: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)

class Error(Base):
    __tablename__ = "errors"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    stage: Mapped[str]
    input: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str] = mapped_column(Text)
    at: Mapped[str]


def migrate(url: str) -> None:
    config = Config()
    config.set_main_option("script_location", str(files("pain_point_agent") / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")

class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path}"
        migrate(url)
        self.engine = create_engine(url)
        event.listen(self.engine, "connect", _enable_foreign_keys)

    def close(self) -> None:
        self.engine.dispose()

    def start_run(self, niche: str, kind: str = "single", parent_id: int | None = None) -> int:
        with Session(self.engine) as session, session.begin():
            run = Run(kind=kind, niche=niche, model=settings.model, started_at=_now(), parent_id=parent_id)
            session.add(run)
            session.flush()
            return run.id

    def error(self, run_id: int, stage: str, error: str, tool_input: dict[str, Any] | None = None) -> None:
        saved_input = json.dumps(tool_input, ensure_ascii=False) if tool_input is not None else None
        with Session(self.engine) as session, session.begin():
            session.add(Error(run_id=run_id, stage=stage, input=saved_input, error=error, at=_now()))

    def tool_call(
        self, run_id: int, name: str, tool_input: dict[str, Any], content: str, is_error: bool, failed: bool = False
    ) -> None:
        if failed:
            self.error(run_id, name, content, tool_input)
            return
        with Session(self.engine) as session, session.begin():
            if is_error:
                session.add(
                    Rejection(run_id=run_id, tool=name, input=json.dumps(tool_input, ensure_ascii=False), reason=content)
                )
            elif name == "search":
                session.add(
                    Search(
                        run_id=run_id,
                        source=tool_input["source"],
                        query=tool_input["query"],
                        results=len(json.loads(content)),
                    )
                )
            elif name == "fetch":
                self._save_thread(session, run_id, tool_input["source"], json.loads(content))

    def finish_run(self, run_id: int, research: Research, report: str) -> None:
        strong = research.strong_pains()
        with Session(self.engine) as session, session.begin():
            run = session.get_one(Run, run_id)
            run.finished_at = _now()
            run.finished = research.finished
            run.stopped = research.stopped
            run.theme = research.theme
            run.note = research.note
            run.consensus = research.consensus
            run.searches = research.searches
            run.errors = session.scalar(select(func.count()).select_from(Error).where(Error.run_id == run_id))
            run.cost_usd = research.usage.get("cost_usd")
            run.usage = json.dumps(research.usage)
            run.report = report
            session.add_all(
                Evidence(
                    run_id=run_id,
                    pain=pain,
                    strong=pain in strong,
                    source=item["source"],
                    url=item["url"],
                    title=item["title"],
                    quote=item["quote"],
                    post_date=item["date"],
                )
                for pain, items in research.pains.items()
                for item in items
            )

    def _save_thread(self, session: Session, run_id: int, source: str, thread: dict[str, Any]) -> None:
        values = {
            "url": thread["url"],
            "title": thread["title"],
            "posts_count": thread.get("posts_count"),
            "last_fetched_at": _now(),
        }
        upsert = insert(Thread).values(source=source, item_id=thread["id"], **values)
        thread_id = session.execute(
            upsert.on_conflict_do_update(index_elements=["source", "item_id"], set_=values).returning(Thread.id)
        ).scalar_one()
        for post in thread["posts"]:
            fields = {"author": post["author"], "created_at": post["created_at"], "text": post["text"]}
            session.execute(
                insert(Post)
                .values(thread_id=thread_id, number=post["number"], **fields)
                .on_conflict_do_update(index_elements=["thread_id", "number"], set_=fields)
            )
        session.execute(insert(RunThread).values(run_id=run_id, thread_id=thread_id).on_conflict_do_nothing())


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _enable_foreign_keys(connection: Any, _: Any) -> None:
    connection.execute("PRAGMA foreign_keys = ON")

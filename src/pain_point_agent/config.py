from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Source(BaseModel):
    type: Literal["discourse", "hn", "github", "appstore"]
    url: str
    about: str
    enabled: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    github_token: SecretStr | None = None
    model: str = "claude-haiku-5-5"
    max_tokens: int = 16000
    max_steps: int = 20
    theme_max_tokens: int = 1000
    judge_max_tokens: int = 300
    judge_votes: int = 1
    judge_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    judge_timeout: float = 30
    price_input_per_mtok: float = 0.10
    price_output_per_mtok: float = 0.50
    cache_read_rate: float = 0.1
    cache_write_rate: float = 1.25

    max_searches: int = 6
    max_searches_per_source: int = 2
    max_source_failures: int = 2
    min_pains: int = 3
    max_pains: int = 5
    min_threads_per_pain: int = 2
    min_quote_chars: int = 20
    max_evidence_age_days: int | None = None

    sources: dict[str, Source] = {
        "n8n": Source(type="discourse", url="https://community.n8n.io", about="workflow automation, AI agents, integrations", enabled=True),
        "openai": Source(type="discourse", url="https://community.openai.com", about="developers building on the OpenAI API", enabled=True),
        "homeassistant": Source(type="discourse", url="https://community.home-assistant.io", about="smart home, home automation", enabled=True),
        "cursor": Source(type="discourse", url="https://forum.cursor.com", about="Cursor users, AI coding assistants", enabled=True),
        "make": Source(type="discourse", url="https://community.make.com", about="Make.com no-code automation", enabled=True),
        "huggingface": Source(type="discourse", url="https://discuss.huggingface.co", about="ML and LLM developers, models, fine-tuning", enabled=True),
        "obsidian": Source(type="discourse", url="https://forum.obsidian.md", about="Obsidian note-taking, personal knowledge management", enabled=True),
        "hn": Source(type="hn", url="https://hn.algolia.com/api/v1", about="Hacker News comments: developers, founders, tech products. Use short 2-4 word queries, long ones return nothing", enabled=True),
        "github": Source(type="github", url="https://api.github.com", about="GitHub issues: bug reports for developer tools and open source libraries. Use short queries", enabled=True),
        "appstore": Source(type="appstore", url="https://itunes.apple.com", about="Apple App Store: search finds apps (posts_count = number of ratings), fetch reads their recent low-rated reviews (posts_count = number of low-rated reviews found). One app is one thread. Query with an app category, e.g. 'habit tracker'", enabled=False),
    }

    user_agent: str = "pain-point-agent"
    http_timeout: float = 15
    max_retries: int = 3
    max_retry_delay: float = 30
    retry_statuses: frozenset[int] = frozenset({429, 502, 503, 504})
    search_limit: int = 10
    max_post_chars: int = 2000
    max_blurb_chars: int = 300
    db_path: Path = Path("data/pain-points.db")
    github_max_comments: int = 30
    appstore_country: str = "us"
    appstore_search_pool: int = 25
    appstore_review_pages: int = 3
    appstore_max_rating: int = 2
    max_thread_chars: int = 8000


settings = Settings()

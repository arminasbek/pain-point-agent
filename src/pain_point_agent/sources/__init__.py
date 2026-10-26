from pain_point_agent.config import settings
from pain_point_agent.sources import appstore, discourse, github, hn
from pain_point_agent.sources.models import SearchResult, Thread

ADAPTERS = {"discourse": discourse, "hn": hn, "github": github, "appstore": appstore}


def search(source: str, query: str) -> list[SearchResult]:
    config = settings.sources[source]
    return ADAPTERS[config.type].search(config.url, query)


def fetch(source: str, item_id: int) -> Thread:
    config = settings.sources[source]
    return ADAPTERS[config.type].fetch(config.url, item_id)

import re
from html.parser import HTMLParser


def html_to_text(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    text = "".join(extractor.parts)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class _TextExtractor(HTMLParser):
    BLOCK_TAGS = {"p", "br", "li", "pre", "blockquote", "div", "h1", "h2", "h3", "h4", "tr"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

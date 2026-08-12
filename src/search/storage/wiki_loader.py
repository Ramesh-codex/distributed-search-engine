import bz2
import re
from collections.abc import Iterator
from xml.etree import ElementTree

# MediaWiki markup stripped before indexing. Left in, "ref", "http" and
# template names become high-frequency index terms that match everything.
_MARKUP_PATTERNS = [
    re.compile(r"\{\{[^}]*\}\}"),
    re.compile(r"<ref[^>]*>.*?</ref>", re.S),
    re.compile(r"<[^>]+>"),
    re.compile(r"\[\[(?:File|Image|Category):[^\]]*\]\]"),
    re.compile(r"https?://\S+"),
    re.compile(r"'{2,}"),
    re.compile(r"={2,}"),
]

# [[target|display]] -> display; [[target]] -> target
_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def _clean(markup: str) -> str:
    text = _WIKILINK.sub(lambda m: m.group(2) or m.group(1), markup)
    for pattern in _MARKUP_PATTERNS:
        text = pattern.sub(" ", text)
    return " ".join(text.split())


def stream_articles(path: str, limit: int | None = None) -> Iterator[tuple[str, str]]:
    """Yield (title, text) from a MediaWiki XML dump.

    iterparse with element clearing, not ElementTree.parse: the uncompressed
    dump is over a gigabyte, and building a DOM would exhaust memory before
    one document was indexed. This holds a single element at a time.
    """
    count = 0
    with bz2.open(path, "rb") as handle:
        context = ElementTree.iterparse(handle, events=("end",))
        for _, element in context:
            if not element.tag.endswith("}page"):
                continue

            ns = element.find("{*}ns")
            title_el = element.find("{*}title")
            text_el = element.find("{*}revision/{*}text")

            # ns 0 is article space. Everything else is Talk:, User:,
            # Template:, Category: -- not content.
            if ns is not None and ns.text == "0" and text_el is not None and text_el.text:
                text = _clean(text_el.text)
                if len(text) > 500:
                    yield title_el.text or "", text
                    count += 1
                    if limit and count >= limit:
                        element.clear()
                        return

            # Without this, the parser retains every element and memory
            # grows linearly with the dump.
            element.clear()

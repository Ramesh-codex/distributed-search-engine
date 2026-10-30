"""Contract checks for the dependency-free portfolio UI, including JS syntax."""
import re
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "src/search/distribution/static/index.html"


class InterfaceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.queries = []
        self.attributes = {}
        self.scripts = []
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
            self.attributes[attrs["id"]] = (tag, attrs)
        if tag == "button" and "data-query" in attrs:
            self.queries.append(attrs["data-query"])
        if tag == "script" and "src" not in attrs:
            self.in_script = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        if self.in_script:
            self.scripts.append(data)


def main() -> None:
    html = PAGE.read_text(encoding="utf-8")
    parser = InterfaceParser()
    parser.feed(html)
    assert len(parser.ids) == len(set(parser.ids)), "duplicate id in markup"

    required = [
        "search-form", "q", "search-button", "meta", "degraded-banner",
        "result-content", "cluster-dot", "cluster-state", "health-note",
        "shard-status-list",
    ]
    assert all(item in parser.ids for item in required), "missing interactive UI element"
    assert parser.attributes["q"][1].get("type") == "search"
    assert "required" in parser.attributes["q"][1], "query should not submit empty"
    assert set(parser.queries) == {
        "circuit breaker", "BM25 ranking", "vector search", "distributed systems"
    }
    assert '<meta name="viewport"' in html
    assert "@media (max-width: 580px)" in html
    assert 'aria-live="polite"' in html
    assert "48 curated technical documents" in html
    assert "127.0.0.1" not in html, "internal loopback URLs must not leak into UI"
    assert "document.createTextNode(result.title" in "".join(parser.scripts)
    assert 'fetch("/search?"' in "".join(parser.scripts)
    assert 'fetch("/stats"' in "".join(parser.scripts)

    js = "\n".join(parser.scripts)
    assert js.strip(), "missing inline JS"
    with tempfile.TemporaryDirectory(prefix="ui-check-") as temp:
        path = Path(temp) / "ui.js"
        path.write_text(js, encoding="utf-8")
        subprocess.run(["node", "--check", str(path)], check=True)
    print("PASS: responsive UI markup, safe result rendering, search controls, JavaScript syntax")


if __name__ == "__main__":
    main()

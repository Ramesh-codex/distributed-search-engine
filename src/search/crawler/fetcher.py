import time
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

USER_AGENT = "distributed-search-engine/0.1 (portfolio project; +https://github.com/Ramesh-codex)"

# 10MB. Without a cap, one linked ISO image stalls the whole crawl.
MAX_BYTES = 10 * 1024 * 1024

# Politeness: one request per host per second. Hammering a server gets your
# IP banned and is the single most common way a naive crawler dies.
HOST_DELAY_SECONDS = 1.0

_HTML_TYPES = ("text/html", "application/xhtml+xml")


class Fetcher:
    """Polite HTTP fetcher: robots.txt aware, rate limited per host.

    State is per-host, not global: a 1s global delay would make a 10-host
    crawl 10x slower than it needs to be, while a per-host delay lets
    different domains proceed in parallel without any of them being hammered.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        self._robots: dict[str, RobotFileParser | None] = {}
        self._last_fetch: dict[str, float] = {}

    def _host(self, url: str) -> str:
        return urlsplit(url).netloc

    def _robots_for(self, url: str) -> RobotFileParser | None:
        """Fetched once per host and cached. A failed fetch caches None,
        which we treat as allow -- the convention when robots.txt is absent."""
        host = self._host(url)
        if host in self._robots:
            return self._robots[host]

        parts = urlsplit(url)
        parser = RobotFileParser()
        try:
            response = self._client.get(f"{parts.scheme}://{host}/robots.txt")
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            else:
                parser = None
        except httpx.HTTPError:
            parser = None

        self._robots[host] = parser
        return parser

    def allowed(self, url: str) -> bool:
        parser = self._robots_for(url)
        return True if parser is None else parser.can_fetch(USER_AGENT, url)

    def _wait_for_host(self, url: str) -> None:
        host = self._host(url)
        last = self._last_fetch.get(host)
        if last is not None:
            elapsed = time.monotonic() - last
            if elapsed < HOST_DELAY_SECONDS:
                time.sleep(HOST_DELAY_SECONDS - elapsed)
        self._last_fetch[host] = time.monotonic()

    def fetch(self, url: str) -> str | None:
        """Returns decoded HTML, or None if disallowed, non-HTML, or failed."""
        if not self.allowed(url):
            return None

        self._wait_for_host(url)

        try:
            response = self._client.get(url)
        except httpx.HTTPError:
            return None

        if response.status_code != 200:
            return None

        # Check type before reading the body: no point downloading a PDF.
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type not in _HTML_TYPES:
            return None

        if len(response.content) > MAX_BYTES:
            return None

        # httpx resolves encoding from the Content-Type header, then falls
        # back to charset detection. Pages that declare one charset in the
        # header and another in a <meta> tag are common; errors="replace"
        # keeps a mis-declared page from killing the crawl.
        return response.content.decode(response.encoding or "utf-8", errors="replace")

    def close(self) -> None:
        self._client.close()

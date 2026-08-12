from collections import deque
from urllib.parse import urldefrag, urlsplit, urlunsplit, parse_qsl, urlencode

# Parameters that identify the referrer or campaign, never the document.
# Without stripping these, one page enters the frontier once per inbound link.
_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "ref", "ref_src",
})

_DEFAULT_PORTS = {"http": "80", "https": "443"}


def canonicalize(url: str) -> str:
    """Collapse URLs that address the same document into one key.

    example.com/page, example.com/page/, EXAMPLE.com/page and
    example.com/page?utm_source=x are all one document. Treating them as
    four fills the index with identically-scoring duplicates.
    """
    url, _ = urldefrag(url)          # #section never identifies a new document
    parts = urlsplit(url)

    scheme = parts.scheme.lower()
    host = parts.hostname or ""

    port = parts.port
    netloc = host
    if port and str(port) != _DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{port}"

    query = urlencode([
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ])

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    return urlunsplit((scheme, netloc, path, query, ""))


class Frontier:
    """BFS crawl queue with canonical deduplication.

    Dedup happens on add, not on pop: a URL linked from 50 pages would
    otherwise sit in the queue 50 times and be fetched 50 times.

    The seen set holds full URL strings, roughly 100 bytes each. Fine to a
    few million; beyond that a Bloom filter trades a small false-positive
    rate (occasionally skipping an uncrawled page) for constant memory.
    """

    def __init__(self, seeds: list[str], max_depth: int = 3) -> None:
        self._queue: deque[tuple[str, int]] = deque()
        self._seen: set[str] = set()
        self._max_depth = max_depth
        for seed in seeds:
            self.add(seed, depth=0)

    def add(self, url: str, depth: int) -> bool:
        """Returns False if already seen or past the depth limit."""
        if depth > self._max_depth:
            return False
        canonical = canonicalize(url)
        if canonical in self._seen:
            return False
        self._seen.add(canonical)
        self._queue.append((canonical, depth))
        return True

    def next(self) -> tuple[str, int] | None:
        """popleft, not pop: FIFO is what makes this breadth-first."""
        return self._queue.popleft() if self._queue else None

    def __len__(self) -> int:
        return len(self._queue)

    @property
    def seen_count(self) -> int:
        return len(self._seen)

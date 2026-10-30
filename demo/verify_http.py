"""Read-only external HTTP verification for a locally or publicly hosted demo."""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request


def read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=8) as response:
        if response.status != 200:
            raise AssertionError(f"unexpected HTTP {response.status}: {url}")
        return json.load(response)


def verify(base: str, wait_seconds: int = 60) -> None:
    base = base.rstrip("/")
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            if read_json(base + "/health") == {"status": "ok"}:
                break
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"health did not become ready at {base}")
        time.sleep(1)

    with urllib.request.urlopen(base + "/", timeout=8) as response:
        html = response.read().decode("utf-8")
        assert response.status == 200
        assert "search-form" in html, "search UI not served"

    stats = read_json(base + "/stats")
    assert len(stats["shard_urls"]) == 3, stats
    assert len(stats["breakers"]) == 3, stats

    for query, expected in (
        ("circuit breaker", "Circuit Breaker Pattern"),
        ("BM25 ranking", "BM25 Relevance Ranking"),
        ("vector search", "Vector Similarity Search"),
    ):
        search_url = base + "/search?" + urllib.parse.urlencode(
            {"q": query, "k": 10}
        )
        payload = read_json(search_url)
        assert payload["shards_queried"] == 3, payload
        assert payload["degraded"] is False, payload
        assert payload["shards_failed"] == [], payload
        assert expected in [r["title"] for r in payload["results"]], payload

    print(f"PASS: {base} serves UI, health, stats, and 3-shard BM25 results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="service origin, no trailing path")
    parser.add_argument("--wait", type=int, default=60)
    args = parser.parse_args()
    verify(args.url, wait_seconds=args.wait)

"""End-to-end smoke test: build indexes, start 4 HTTP processes, query them."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=4) as response:
        assert response.status == 200
        return json.load(response)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="search-demo-") as tmp:
        index_dir = Path(tmp) / "index"
        subprocess.run(
            [sys.executable, str(ROOT / "demo" / "build_demo.py"),
             "--out", str(index_dir)],
            check=True, cwd=ROOT
        )
        meta = json.loads((index_dir / "cluster.json").read_text())
        assert meta["global_document_count"] == 48
        public_port = free_port()
        shard_base_port = 18080
        env = os.environ.copy()
        env["DEMO_INDEX_DIR"] = str(index_dir)
        env["PORT"] = str(public_port)
        env["DEMO_SHARD_BASE_PORT"] = str(shard_base_port)
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "demo" / "run_cluster.py")],
            env=env, cwd=ROOT
        )
        try:
            base = f"http://127.0.0.1:{public_port}"
            deadline = time.monotonic() + 35
            while True:
                if process.poll() is not None:
                    raise AssertionError("cluster exited before ready")
                try:
                    assert get_json(base + "/health") == {"status": "ok"}
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("coordinator was not ready")
                    time.sleep(0.2)

            with urllib.request.urlopen(base + "/", timeout=4) as response:
                html = response.read().decode()
                assert "<html" in html.lower()
                assert "search-form" in html
            stats = get_json(base + "/stats")
            assert len(stats["shard_urls"]) == 3
            assert all(item["state"] == "closed" for item in stats["breakers"].values())
            for query, expected_title in [
                ("circuit breaker", "Circuit Breaker Pattern"),
                ("BM25 ranking", "BM25 Relevance Ranking"),
                ("vector search", "Vector Similarity Search"),
            ]:
                result = get_json(base + "/search?" + urllib.parse.urlencode({
                    "q": query, "k": 10,
                }))
                assert result["shards_queried"] == 3, result
                assert result["shards_failed"] == [], result
                assert result["degraded"] is False, result
                assert result["results"], result
                assert expected_title in [r["title"] for r in result["results"]], result

            no_results = get_json(base + "/search?q=zzzznonexistenttoken")
            assert not no_results["results"]
            try:
                urllib.request.urlopen(base + "/search", timeout=4)
            except urllib.error.HTTPError as exc:
                assert exc.code == 422
            else:
                raise AssertionError("missing query should return 422")
            print("PASS: UI, health, stats, 3 live shards, BM25 queries, empty/missing queries")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()

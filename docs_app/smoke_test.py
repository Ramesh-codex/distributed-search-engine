"""Integration test for the real three-shard documentation search."""
import json, os, socket, subprocess, sys, tempfile, time
import urllib.error, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        assert response.status == 200
        return response.read()

def main():
    with tempfile.TemporaryDirectory() as temp:
        idx = Path(temp) / "idx"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
        subprocess.run([sys.executable, "-m", "docs_app.build_index",
                        "--out", str(idx)], env=env, cwd=ROOT, check=True)
        meta = json.loads((idx / "documents.json").read_text())
        assert meta["document_count"] == 188, meta
        assert meta["source_pages"] == 20, meta
        assert meta["source_counts"] == {"FastAPI":86,"HTTPX":54,"Docker":48}
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env.update({"DEMO_INDEX_DIR":str(idx),
                    "DOCS_METADATA_FILE":str(idx / "documents.json"),
                    "DEMO_SHARD_BASE_PORT":"18220","PORT":str(port)})
        proc = subprocess.Popen([sys.executable,"-m","docs_app.run_cluster"],
                                env=env,cwd=ROOT)
        root = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic()+40
            while True:
                assert proc.poll() is None, "Cluster exited early"
                try:
                    assert json.loads(get(root+"/health")) == {"status":"ok"}
                    break
                except (OSError,urllib.error.URLError):
                    if time.monotonic()>deadline: raise
                    time.sleep(.3)
            html = get(root+"/").decode()
            assert "DevDocs Search" in html and "search-form" in html
            assert "data-source" in html
            stats=json.loads(get(root+"/stats"))
            assert len(stats["shard_urls"])==3
            assert all(x["state"]=="closed" for x in stats["breakers"].values())
            assert stats["document_count"]==188
            for query, source, domain in [
                ("dependency","FastAPI","fastapi.tiangolo.com"),
                ("timeout","HTTPX","www.python-httpx.org"),
                ("Dockerfile","Docker","docs.docker.com"),
            ]:
                params=urllib.parse.urlencode({"q":query,"source":source,"k":5})
                r=json.loads(get(root+"/search?"+params))
                assert r["results"] and not r["degraded"],r
                assert r["shards_queried"]==3 and not r["shards_failed"],r
                for hit in r["results"]:
                    assert hit["source"]==source and domain in hit["url"],hit
                    assert hit["snippet"] and hit["title"] and hit["score"]>0,hit
            assert not json.loads(get(root+"/search?q=zzzz_unfindable_xyz"))["results"]
            for params in ["","?q=","?q=test&source=invalid","?q=test&k=1000"]:
                try:
                    get(root+"/search"+params)
                except urllib.error.HTTPError as exc:
                    assert exc.code==422
                else: raise AssertionError(f"Expected HTTP 422 for {params}")
            assert "swagger-ui" in get(root+"/docs").decode()
            print("PASS: docs UI, 188 indexed sections, 3 shard HTTP searches, filters, source URLs, API validation")
        finally:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill();proc.wait()

if __name__ == "__main__":
    main()

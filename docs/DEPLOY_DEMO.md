# Render portfolio demo

This demo deploys **the real three-shard HTTP search architecture** with a
small, original 48-document technical glossary instead of the original 30K
Wikipedia benchmark corpus. **Do not compare demo latency to README benchmark
numbers**: the demo corpus and hosting hardware are different.

## What is deployed

- A container image builds the immutable three-shard index at image-build time.
- `demo/run_cluster.py` starts three independent FastAPI shard processes on
  localhost ports 18080 through 18082 and waits for their health endpoints.
- It starts the existing FastAPI HTTP coordinator on Render's `$PORT`.
- `GET /` serves the original coordinator search UI. `/search?q=circuit%20breaker`,
  `/stats`, `/health`, and `/docs` are reachable through the same public URL.
- The existing `Dockerfile`, `docker-compose.yml`, and `nginx.conf`
  remain unchanged; nginx is used by the local Compose version, not the
  one-container free-hosting demo.

This is a **cost-optimized demonstration**: the shard processes share one
instance, so it does not demonstrate cross-machine fault isolation or scaling.
The shards still communicate with the coordinator over real HTTP.

## Test

Run from the repository root:

```bash
python -m pip install -r requirements-demo.txt
PYTHONPATH=src python demo/smoke_test.py
```

The test builds the demo indexes in a temporary folder, starts three shard
servers and the coordinator, sends HTTP requests through the public-facing
coordinator, checks the UI, three healthy shard URLs, BM25 hits, and negative
cases, then shuts down the processes. It does not require Docker, network
downloads, or the Wikipedia dump.

For a Docker test, build and run locally:

```bash
docker build -f Dockerfile.render -t search-demo .
docker run --rm -p 10000:10000 -e PORT=10000 search-demo
```

Visit http://localhost:10000 and try `circuit breaker`, `BM25 ranking`,
`distributed search`, or `vector search`.

## Deploy through Render

1. In Render, choose **New > Blueprint** and connect this GitHub repository.
2. Choose the `deployment-demo` branch containing `render.yaml`.
3. Review the Blueprint: **one web service on plan `free`**. Do not accept
   upgrades to paid plans unless you choose to.
4. Apply the Blueprint, then wait for the build and health checks to succeed.
5. Open the actual `https://...onrender.com` URL shown by Render and run
   `python demo/verify_http.py --url https://YOUR-RENDER-URL.onrender.com`.
   This checks `/`, `/health`, `/stats`, and real ranked HTTP search.
6. Add the *verified* URL to the README only after the live test passes.

Free Render services may spin down after 15 minutes without requests, and
cold starts may take about a minute. The demo index is baked into the image
to survive restarts. Do not claim 24/7 production availability.

No paid products, storage volumes, databases, or API keys are required.

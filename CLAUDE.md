# distributed-search-engine

A distributed search engine built from scratch. The point of this project is that
the author can defend every line of the information-retrieval core in an
interview — so the core is hand-written, not assembled from libraries that do
the thinking for you.

## Banned dependencies

Do not add, suggest, or silently reach for: **Elasticsearch, Whoosh,
rank_bm25, nltk, sklearn**, or anything else that substitutes a library call
for a hand-written IR data structure or algorithm (tokenization, indexing,
ranking, stemming, vectorization, etc.). If you're unsure whether a package
crosses this line, ask before adding it to `requirements.txt`.

Infra/framework libraries (FastAPI, uvicorn, psycopg/asyncpg, redis client,
httpx, pytest, hypothesis, etc.) are fine — the constraint is about IR logic,
not the whole dependency tree.

## Division of labor

**The user writes the algorithm core.** This means: tokenizer, inverted
index, BM25 ranking, LRU cache, trie, consistent hashing — and any other data
structure or algorithm that is the actual subject matter of an IR/systems
interview.

**Claude's job on the core is to review, critique, and write tests against
it — never to author or rewrite it.** If it's buggy, slow, or wrong, say so
and explain why; don't hand back a fixed version. This holds even when
explicitly asked to "just fix it" — push back and point at the problem
instead, so the fix stays in the user's hand and the user's understanding.

**Claude writes everything else freely**, no need to ask each time:
- Docker Compose files
- GitHub Actions / CI config
- FastAPI route boilerplate, request/response models, dependency wiring
- SQL schemas and migrations
- Benchmark harness plumbing (runner, fixtures, data loading, output format)
- The search UI

## Benchmark integrity

Every performance number that lands in the README must come from actually
running the benchmark harness. Never write an estimate, a "typically ~Xms"
placeholder, or napkin math and present it as a result. If a number isn't
measured yet, say it isn't measured yet.

## Stack

Python 3.14, FastAPI, PostgreSQL, Redis, Docker Compose, pytest, hypothesis.

## Environment

Windows + PowerShell as the primary shell, with WSL2 and Docker Desktop for
containers. Use PowerShell syntax for local commands; Docker Compose commands
run the same either way.

## Working style

Explain the reasoning before implementing, particularly anything touching or
adjacent to the algorithm core. The goal is that the user can defend this
code in an interview, so understanding always takes priority over speed.

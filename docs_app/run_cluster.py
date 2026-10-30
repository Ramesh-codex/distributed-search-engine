"""Launch the original three real shard servers with the docs search coordinator."""
import os

os.environ.setdefault("DEMO_COORDINATOR_MODULE", "docs_app.server:app")

from demo.run_cluster import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

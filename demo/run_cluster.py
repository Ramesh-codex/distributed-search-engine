"""Run the three real shard APIs and HTTP coordinator in one demo container.

This is a low-cost portfolio deployment layout, not a recommendation to
co-locate distributed services in production. The local Compose setup remains
available for independently deployed containers and nginx.
"""
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

INDEX_DIR = Path(os.environ.get("DEMO_INDEX_DIR", "/app/demo-index"))
BASE_PORT = int(os.environ.get("DEMO_SHARD_BASE_PORT", "18080"))
PUBLIC_PORT = int(os.environ.get("PORT", "10000"))
PROCESSES = []
STOPPING = False


def stop(*_args):
    global STOPPING
    STOPPING = True
    for proc in reversed(PROCESSES):
        if proc.poll() is None:
            proc.terminate()


def await_shard(proc, port: int, seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not STOPPING:
        if proc.poll() is not None:
            raise RuntimeError(f"shard on {port} exited early ({proc.returncode})")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=1.0
            ) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"shard on port {port} did not become healthy")


def start_process(module: str, port: int, host: str, env: dict) -> subprocess.Popen:
    proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", module,
        "--host", host, "--port", str(port), "--workers", "1"
    ], env=env)
    PROCESSES.append(proc)
    return proc


def main() -> int:
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if not (INDEX_DIR / "cluster.json").is_file():
        raise FileNotFoundError(f"prebuilt demo index missing: {INDEX_DIR}")

    try:
        shard_urls = []
        for i in range(3):
            shard_dir = INDEX_DIR / f"shard-{i}"
            if not (shard_dir / "index.globals").is_file():
                raise FileNotFoundError(f"missing shard index: {shard_dir}")
            port = BASE_PORT + i
            env = os.environ.copy()
            env["SHARD_DIR"] = str(shard_dir)
            process = start_process(
                "search.distribution.shard_server:app",
                port, "127.0.0.1", env
            )
            await_shard(process, port)
            shard_urls.append(f"http://127.0.0.1:{port}")

        env = os.environ.copy()
        env["SHARD_URLS"] = ",".join(shard_urls)
        env.setdefault("SHARD_CONNECT_TIMEOUT", "0.4")
        env.setdefault("SHARD_READ_TIMEOUT", "2.0")
        coordinator = start_process(
            os.environ.get("DEMO_COORDINATOR_MODULE", "search.distribution.coordinator_server:app"),
            PUBLIC_PORT, "0.0.0.0", env
        )
        print(f"Serving three-shard search on port {PUBLIC_PORT}", flush=True)
        while not STOPPING:
            if coordinator.poll() is not None:
                return coordinator.returncode or 1
            # If a shard crashes, fail the container rather than hiding it.
            if any(proc.poll() is not None for proc in PROCESSES[:3]):
                raise RuntimeError("a shard process exited unexpectedly")
            time.sleep(0.25)
        return 0
    finally:
        stop()
        for proc in reversed(PROCESSES):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())

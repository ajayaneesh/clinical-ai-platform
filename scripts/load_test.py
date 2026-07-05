"""Load simulation to establish a performance baseline for /predict.

Fires N requests at a running server and reports throughput + latency
percentiles. Run against a REAL server (not in-process) so timings include the
uvicorn/network stack you'd see in production:

    # terminal 1
    uv run serve
    # terminal 2
    uv run python scripts/load_test.py --requests 100 500 1000

Metrics gathered here (client-side latency, throughput) complement the
server-side Prometheus metrics at /metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from base64 import b64encode

import httpx

VALID_IMAGE = b64encode(b"fake image bytes").decode()


async def _one_request(client: httpx.AsyncClient, url: str) -> tuple[int, float]:
    start = time.perf_counter()
    resp = await client.post(url, json={"image": VALID_IMAGE})
    elapsed = time.perf_counter() - start
    return resp.status_code, elapsed


async def run_batch(base_url: str, total: int, concurrency: int) -> dict[str, float]:
    url = f"{base_url}/predict"
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    statuses: list[int] = []

    async with httpx.AsyncClient(timeout=30.0) as client:

        async def bounded() -> None:
            async with semaphore:
                status, elapsed = await _one_request(client, url)
                statuses.append(status)
                latencies.append(elapsed)

        wall_start = time.perf_counter()
        await asyncio.gather(*(bounded() for _ in range(total)))
        wall = time.perf_counter() - wall_start

    ok = sum(1 for s in statuses if s == 200)
    latencies_ms = sorted(x * 1000 for x in latencies)

    def pct(p: float) -> float:
        # nearest-rank percentile
        idx = min(len(latencies_ms) - 1, int(p / 100 * len(latencies_ms)))
        return latencies_ms[idx]

    return {
        "requests": total,
        "concurrency": concurrency,
        "wall_seconds": round(wall, 3),
        "throughput_rps": round(total / wall, 1) if wall else 0.0,
        "success": ok,
        "failed": total - ok,
        "p50_ms": round(statistics.median(latencies_ms), 2),
        "p95_ms": round(pct(95), 2),
        "p99_ms": round(pct(99), 2),
        "max_ms": round(latencies_ms[-1], 2),
    }


def _print_report(rows: list[dict[str, float]]) -> None:
    cols = [
        "requests",
        "concurrency",
        "wall_seconds",
        "throughput_rps",
        "success",
        "failed",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
    ]
    header = " | ".join(f"{c:>14}" for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(f"{row[c]:>14}" for c in cols))


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Load-test /predict for a baseline.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()

    rows = []
    for total in args.requests:
        print(f"running {total} requests (concurrency={args.concurrency})...")
        rows.append(await run_batch(args.base_url, total, args.concurrency))
    print()
    _print_report(rows)


if __name__ == "__main__":
    asyncio.run(_main())

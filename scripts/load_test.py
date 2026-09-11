"""Small reproducible concurrent HTTP benchmark; reports measured results only."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/recommend")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--user-id", default="1")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        raise SystemExit("--requests and --concurrency must be positive")

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    semaphore = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    successes = 0

    async with httpx.AsyncClient(timeout=60) as client:

        async def request_once() -> None:
            nonlocal successes
            async with semaphore:
                started = time.perf_counter()
                response = await client.post(
                    args.url,
                    json={"user_id": args.user_id, "top_n": 5},
                    headers=headers,
                )
                latencies.append(time.perf_counter() - started)
                if response.is_success:
                    successes += 1

        started = time.perf_counter()
        await asyncio.gather(*(request_once() for _ in range(args.requests)))
        elapsed = time.perf_counter() - started

    ordered = sorted(latencies)

    def percentile(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(len(ordered) * p))]

    print(f"requests={args.requests}")
    print(f"concurrency={args.concurrency}")
    print(f"duration_seconds={elapsed:.3f}")
    print(f"throughput_rps={args.requests / elapsed:.2f}")
    print(f"success_rate={successes / args.requests:.3f}")
    print(f"mean_latency_ms={statistics.mean(latencies) * 1000:.2f}")
    print(f"p50_latency_ms={percentile(0.50) * 1000:.2f}")
    print(f"p95_latency_ms={percentile(0.95) * 1000:.2f}")
    print(f"p99_latency_ms={percentile(0.99) * 1000:.2f}")


if __name__ == "__main__":
    asyncio.run(main())

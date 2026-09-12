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
    parser.add_argument(
        "--rate-limit",
        action="store_true",
        help="Explicitly test rate limit behavior (expect 429 responses)",
    )
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        raise SystemExit("--requests and --concurrency must be positive")

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    semaphore = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    successes = 0
    rate_limited = 0
    other_http_errors = 0
    connect_errors = 0
    timeout_errors = 0
    request_errors = 0

    async with httpx.AsyncClient(timeout=60) as client:

        async def request_once() -> None:
            nonlocal successes, rate_limited, other_http_errors, connect_errors
            nonlocal timeout_errors, request_errors
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post(
                        args.url,
                        json={"user_id": args.user_id, "top_n": 5},
                        headers=headers,
                    )
                    latencies.append(time.perf_counter() - started)
                    if response.is_success:
                        successes += 1
                    elif response.status_code == 429:
                        rate_limited += 1
                    else:
                        other_http_errors += 1
                except httpx.ConnectError:
                    connect_errors += 1
                    latencies.append(
                        time.perf_counter() - started
                    )  # Still measure time
                except httpx.TimeoutException:
                    timeout_errors += 1
                    latencies.append(
                        time.perf_counter() - started
                    )  # Still measure time
                except httpx.HTTPError:
                    request_errors += 1
                    latencies.append(
                        time.perf_counter() - started
                    )  # Still measure time
                except Exception:
                    request_errors += 1
                    latencies.append(
                        time.perf_counter() - started
                    )  # Still measure time

        started = time.perf_counter()
        await asyncio.gather(*(request_once() for _ in range(args.requests)))
        elapsed = time.perf_counter() - started

    ordered = sorted(latencies)

    def percentile(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(len(ordered) * p))]

    print(f"requests={args.requests}")
    print(f"concurrency={args.concurrency}")
    if args.rate_limit:
        print("rate_limit_test=true")
    print(f"duration_seconds={elapsed:.3f}")
    print(f"throughput_rps={args.requests / elapsed:.2f}")
    print(f"success_rate={successes / args.requests:.3f}")
    print(f"rate_limit_rate={rate_limited / args.requests:.3f}")
    print(f"other_http_error_rate={other_http_errors / args.requests:.3f}")
    print(f"connect_error_rate={connect_errors / args.requests:.3f}")
    print(f"timeout_error_rate={timeout_errors / args.requests:.3f}")
    print(f"request_error_rate={request_errors / args.requests:.3f}")
    print(f"mean_latency_ms={statistics.mean(latencies) * 1000:.2f}")
    print(f"p50_latency_ms={percentile(0.50) * 1000:.2f}")
    print(f"p95_latency_ms={percentile(0.95) * 1000:.2f}")
    print(f"p99_latency_ms={percentile(0.99) * 1000:.2f}")


if __name__ == "__main__":
    asyncio.run(main())

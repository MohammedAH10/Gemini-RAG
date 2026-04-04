"""
Benchmark script for query performance.
"""

import statistics
import time
from typing import Dict, List

import requests

BASE_URL = "http://localhost:8000/api/v1"


def benchmark_query(token: str, query: str, n_iterations: int = 10) -> Dict:
    """
    Benchmark a single query.

    Args:
        token: Access token
        query: Query text
        n_iterations: Number of iterations

    Returns:
        Benchmark results
    """
    times = []

    for i in range(n_iterations):
        start = time.time()

        response = requests.post(
            f"{BASE_URL}/query/ask",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": query, "n_results": 5},
        )

        elapsed = time.time() - start
        times.append(elapsed)

        if response.status_code != 200:
            print(f"Error on iteration {i}: {response.status_code}")

    return {
        "query": query,
        "iterations": n_iterations,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
    }


def run_benchmarks(token: str):
    """Run complete benchmark suite."""
    queries = [
        "What is machine learning?",
        "Explain artificial intelligence",
        "What is deep learning?",
        "How do neural networks work?",
        "What is natural language processing?",
    ]

    print("=== Query Performance Benchmarks ===\n")

    for query in queries:
        print(f"Benchmarking: {query}")
        results = benchmark_query(token, query, n_iterations=5)

        print(f"  Mean: {results['mean']:.3f}s")
        print(f"  Median: {results['median']:.3f}s")
        print(f"  Min: {results['min']:.3f}s")
        print(f"  Max: {results['max']:.3f}s")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python benchmark_queries.py <access_token>")
        sys.exit(1)

    token = sys.argv[1]
    run_benchmarks(token)

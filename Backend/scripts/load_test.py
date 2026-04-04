"""
Load testing script for RAG system.
"""
import asyncio
import aiohttp
import time
from typing import List, Dict
import statistics

BASE_URL = "http://localhost:8000/api/v1"


async def create_user(session: aiohttp.ClientSession, user_num: int) -> Dict:
    """Create a test user."""
    email = f"loadtest-{user_num}-{int(time.time())}@example.com"

    async with session.post(
        f"{BASE_URL}/auth/signup",
        json={
            "email": email,
            "password": "LoadTest123!"
        }
    ) as response:
        data = await response.json()
        return {
            "email": email,
            "token": data.get("access_token"),
            "user_num": user_num
        }


async def upload_document(session: aiohttp.ClientSession, user: Dict) -> str:
    """Upload a document."""
    content = f"Test document for user {user['user_num']}. Machine learning content."

    data = aiohttp.FormData()
    data.add_field('file', content, filename='test.txt', content_type='text/plain')
    data.add_field('title', f"Doc {user['user_num']}")

    async with session.post(
        f"{BASE_URL}/documents/upload",
        headers={"Authorization": f"Bearer {user['token']}"},
        data=data
    ) as response:
        result = await response.json()
        return result.get("document_id")


async def make_query(session: aiohttp.ClientSession, user: Dict) -> float:
    """Make a query and return response time."""
    start = time.time()

    async with session.post(
        f"{BASE_URL}/query/ask",
        headers={"Authorization": f"Bearer {user['token']}"},
        json={
            "query": "What is machine learning?",
            "n_results": 3
        }
    ) as response:
        await response.json()
        return time.time() - start


async def user_workflow(session: aiohttp.ClientSession, user_num: int) -> Dict:
    """Complete user workflow."""
    try:
        # Create user
        user = await create_user(session, user_num)

        # Upload document
        await upload_document(session, user)

        # Wait for indexing
        await asyncio.sleep(2)

        # Make multiple queries
        query_times = []
        for _ in range(5):
            duration = await make_query(session, user)
            query_times.append(duration)

        return {
            "user_num": user_num,
            "success": True,
            "avg_query_time": statistics.mean(query_times),
            "query_times": query_times
        }

    except Exception as e:
        return {
            "user_num": user_num,
            "success": False,
            "error": str(e)
        }


async def run_load_test(num_users: int = 100):
    """Run load test with multiple concurrent users."""
    print(f"=== Load Test: {num_users} Concurrent Users ===\n")

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        # Create all user workflows
        tasks = [user_workflow(session, i) for i in range(num_users)]

        # Run concurrently
        results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if successful:
        all_query_times = [t for r in successful for t in r["query_times"]]

        print(f"Total Users: {num_users}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"\nQuery Performance:")
        print(f"  Mean: {statistics.mean(all_query_times):.3f}s")
        print(f"  Median: {statistics.median(all_query_times):.3f}s")
        print(f"  Min: {min(all_query_times):.3f}s")
        print(f"  Max: {max(all_query_times):.3f}s")
        print(f"  StdDev: {statistics.stdev(all_query_times):.3f}s")

    if failed:
        print(f"\nFailed Users:")
        for fail in failed[:5]:  # Show first 5
            print(f"  User {fail['user_num']}: {fail['error']}")


if __name__ == "__main__":
    import sys

    num_users = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    asyncio.run(run_load_test(num_users))

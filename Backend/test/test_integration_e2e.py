"""
Sprint 12: Complete Integration and E2E Tests
"""

import os
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Set environment variables
os.environ["SECRET_KEY"] = "test_secret_key_minimum_32_characters_long"
os.environ["GEMINI_API_KEY"] = "AIzaSyTestKey123456789"
os.environ["SUPABASE_URL"] = "https://test.supabase.co"
os.environ["SUPABASE_KEY"] = "test_key_123456"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test_service_key_123456"

from app.main import app

client = TestClient(app)


@pytest.fixture
def test_user():
    """Create a test user."""
    email = f"test-{int(time.time())}@example.com"
    password = "TestPass123!"

    # Sign up
    response = client.post(
        "/api/v1/auth/signup", json={"email": email, "password": password}
    )

    assert response.status_code == 200
    data = response.json()

    return {
        "email": email,
        "password": password,
        "token": data.get("access_token"),
        "user_id": data.get("user_id"),
    }


@pytest.fixture
def test_document(test_user):
    """Upload a test document."""
    # Create temp file
    content = """
    Machine learning is a subset of artificial intelligence.
    Deep learning uses neural networks to process data.
    Natural language processing enables computers to understand human language.
    """

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    temp_file.write(content)
    temp_file.close()

    # Upload
    with open(temp_file.name, "rb") as f:
        response = client.post(
            "/api/v1/documents/upload",
            headers={"Authorization": f"Bearer {test_user['token']}"},
            files={"file": ("test.txt", f, "text/plain")},
            data={"title": "Test Document"},
        )

    assert response.status_code == 201
    doc_data = response.json()

    # Cleanup
    Path(temp_file.name).unlink()

    return doc_data


# Complete User Journey Tests
def test_complete_user_journey():
    """
    Test: register → login → upload → index → query → logout
    """
    # Step 1: Register
    email = f"journey-{int(time.time())}@example.com"
    signup_response = client.post(
        "/api/v1/auth/signup", json={"email": email, "password": "JourneyPass123!"}
    )

    assert signup_response.status_code == 200
    signup_data = signup_response.json()
    assert signup_data["success"]
    token = signup_data["access_token"]

    # Step 2: Login
    login_response = client.post(
        "/api/v1/auth/signin", json={"email": email, "password": "JourneyPass123!"}
    )

    assert login_response.status_code == 200
    assert login_response.json()["success"]

    # Step 3: Upload document
    content = "Artificial intelligence is transforming technology."
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    temp_file.write(content)
    temp_file.close()

    with open(temp_file.name, "rb") as f:
        upload_response = client.post(
            "/api/v1/documents/upload",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("ai.txt", f, "text/plain")},
            data={"title": "AI Guide"},
        )

    assert upload_response.status_code == 201
    doc_id = upload_response.json()["document_id"]

    # Step 4: Wait for indexing
    time.sleep(2)

    # Step 5: Query
    query_response = client.post(
        "/api/v1/query/ask",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "What is artificial intelligence?", "n_results": 3},
    )

    assert query_response.status_code == 200
    query_data = query_response.json()
    assert query_data["success"]
    assert len(query_data["answer"]) > 0

    # Step 6: Get user info
    user_response = client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )

    assert user_response.status_code == 200

    # Step 7: Logout
    logout_response = client.post(
        "/api/v1/auth/signout", headers={"Authorization": f"Bearer {token}"}
    )

    assert logout_response.status_code == 200

    # Cleanup
    Path(temp_file.name).unlink()


def test_multi_user_isolation(test_user):
    """
    Test that users cannot access each other's data.
    """
    # Create second user
    email2 = f"user2-{int(time.time())}@example.com"
    signup2 = client.post(
        "/api/v1/auth/signup", json={"email": email2, "password": "User2Pass123!"}
    )

    user2_token = signup2.json()["access_token"]

    # User 1 uploads document
    content = "User 1's private data"
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    temp_file.write(content)
    temp_file.close()

    with open(temp_file.name, "rb") as f:
        upload1 = client.post(
            "/api/v1/documents/upload",
            headers={"Authorization": f"Bearer {test_user['token']}"},
            files={"file": ("private.txt", f, "text/plain")},
        )

    doc_id = upload1.json()["document_id"]

    # User 2 tries to access User 1's document
    access_response = client.get(
        f"/api/v1/documents/{doc_id}",
        headers={"Authorization": f"Bearer {user2_token}"},
    )

    # Should be forbidden or not found
    assert access_response.status_code in [403, 404]

    # User 2 queries - should not get User 1's data
    query_response = client.post(
        "/api/v1/query/ask",
        headers={"Authorization": f"Bearer {user2_token}"},
        json={"query": "private data", "n_results": 5},
    )

    # Should get no results or generic answer
    query_data = query_response.json()
    assert query_data["chunks_retrieved"] == 0

    # Cleanup
    Path(temp_file.name).unlink()


def test_concurrent_requests(test_user, test_document):
    """
    Test concurrent query handling.
    """
    import concurrent.futures

    def make_query(i):
        response = client.post(
            "/api/v1/query/ask",
            headers={"Authorization": f"Bearer {test_user['token']}"},
            json={"query": f"Test query {i}", "n_results": 3},
        )
        return response.status_code == 200

    # Make 10 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_query, i) for i in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All should succeed
    assert all(results)


def test_error_recovery():
    """
    Test system recovery from errors.
    """
    # Test with invalid token
    response = client.post(
        "/api/v1/query/ask",
        headers={"Authorization": "Bearer invalid_token"},
        json={"query": "test"},
    )

    assert response.status_code == 401

    # Test with missing auth
    response = client.get("/api/v1/documents")
    assert response.status_code in [401, 403]

    # Test with malformed JSON
    response = client.post("/api/v1/auth/signin", data="not json")
    assert response.status_code == 422


def test_rate_limiting(test_user):
    """
    Test rate limiting enforcement.
    """
    # Make many requests rapidly
    responses = []

    for i in range(25):  # Exceed query limit (20/hour)
        response = client.post(
            "/api/v1/query/ask",
            headers={"Authorization": f"Bearer {test_user['token']}"},
            json={"query": f"Query {i}", "n_results": 1},
        )
        responses.append(response.status_code)

    # Should eventually get rate limited
    assert 429 in responses


def test_pagination(test_user):
    """
    Test document list pagination.
    """
    # Upload multiple documents
    for i in range(5):
        content = f"Document {i} content"
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        temp_file.write(content)
        temp_file.close()

        with open(temp_file.name, "rb") as f:
            client.post(
                "/api/v1/documents/upload",
                headers={"Authorization": f"Bearer {test_user['token']}"},
                files={"file": (f"doc{i}.txt", f, "text/plain")},
            )

        Path(temp_file.name).unlink()

    # Get page 1
    response = client.get(
        "/api/v1/documents?page=1&page_size=3",
        headers={"Authorization": f"Bearer {test_user['token']}"},
    )

    data = response.json()
    assert data["page"] == 1
    assert len(data["documents"]) <= 3
    assert data["total"] >= 5


def test_query_performance(test_user, test_document):
    """
    Test query response time is acceptable.
    """
    start = time.time()

    response = client.post(
        "/api/v1/query/ask",
        headers={"Authorization": f"Bearer {test_user['token']}"},
        json={"query": "What is machine learning?", "n_results": 5},
    )

    duration = time.time() - start

    assert response.status_code == 200
    assert duration < 5.0  # Should complete in under 5 seconds


def test_document_lifecycle(test_user):
    """
    Test complete document lifecycle: upload → view → update → delete.
    """
    # Upload
    content = "Test document content"
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    temp_file.write(content)
    temp_file.close()

    with open(temp_file.name, "rb") as f:
        upload_resp = client.post(
            "/api/v1/documents/upload",
            headers={"Authorization": f"Bearer {test_user['token']}"},
            files={"file": ("lifecycle.txt", f, "text/plain")},
            data={"title": "Lifecycle Test"},
        )

    doc_id = upload_resp.json()["document_id"]

    # View
    view_resp = client.get(
        f"/api/v1/documents/{doc_id}",
        headers={"Authorization": f"Bearer {test_user['token']}"},
    )

    assert view_resp.status_code == 200

    # Update
    update_resp = client.patch(
        f"/api/v1/documents/{doc_id}",
        headers={"Authorization": f"Bearer {test_user['token']}"},
        json={"title": "Updated Title", "tags": ["updated"]},
    )

    assert update_resp.status_code == 200

    # Delete
    delete_resp = client.delete(
        f"/api/v1/documents/{doc_id}",
        headers={"Authorization": f"Bearer {test_user['token']}"},
    )

    assert delete_resp.status_code == 200

    # Verify deleted
    verify_resp = client.get(
        f"/api/v1/documents/{doc_id}",
        headers={"Authorization": f"Bearer {test_user['token']}"},
    )

    assert verify_resp.status_code == 404

    # Cleanup
    Path(temp_file.name).unlink()


def test_health_endpoints():
    """
    Test health check endpoints.
    """
    # App health
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

    # Query health
    response = client.get("/api/v1/query/health")
    assert response.status_code == 200
    assert "components" in response.json()

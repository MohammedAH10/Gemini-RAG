# """
# Sprint 6 Tests: Retrieval System
# """
# import pytest
# import time
# from unittest.mock import Mock, patch
# from fastapi.testclient import TestClient

# from app.main import app
# from app.services.query_service import QueryService, QueryServiceError
# from app.core.vector_store import VectorStore
# from app.core.embeddings import EmbeddingService


# # --- Unit Tests for QueryService ---

# @pytest.fixture
# def mock_embedding_service():
#     service = Mock(spec=EmbeddingService)
#     service.generate_query_embedding.return_value = [0.1] * 768
#     return service


# @pytest.fixture
# def mock_vector_store():
#     store = Mock(spec=VectorStore)
#     store.query.return_value = {
#         "ids": ["chunk1", "chunk2", "chunk3"],
#         "documents": ["doc1 text", "doc2 text", "doc3 text"],
#         "metadatas": [
#             {"document_id": "docA", "user_id": "user1"},
#             {"document_id": "docA", "user_id": "user1"},
#             {"document_id": "docB", "user_id": "user2"},
#         ],
#         "distances": [0.1, 0.3, 0.5],
#     }
#     return store


# @pytest.fixture
# def query_service(mock_embedding_service, mock_vector_store):
#     return QueryService(
#         embedding_service=mock_embedding_service,
#         vector_store=mock_vector_store
#     )


# def test_search_returns_top_k(query_service):
#     """Retrieve top-k chunks for a query."""
#     result = query_service.search("test query", top_k=5)
#     assert len(result["results"]) == 3
#     assert result["total_results"] == 3
#     # Check ordering (first result has highest similarity)
#     assert result["results"][0].chunk_id == "chunk1"
#     assert result["results"][0].similarity > result["results"][1].similarity


# def test_search_with_min_similarity_filter(query_service):
#     """Test with similarity threshold."""
#     # Distances: 0.1 -> sim=1/(1+0.1)=0.909, 0.3->0.769, 0.5->0.667
#     result = query_service.search("test", min_similarity=0.8)
#     assert len(result["results"]) == 1
#     assert result["results"][0].chunk_id == "chunk1"
#     assert result["results"][0].similarity > 0.8


# def test_filter_by_user_id(query_service, mock_vector_store):
#     """Filter results by user_id."""
#     result = query_service.search("test", user_id="user1")
#     # Verify that vector_store.query was called with correct user_id
#     mock_vector_store.query.assert_called_with(
#         query_embedding=[0.1]*768,
#         n_results=5,
#         user_id="user1",
#         document_id=None,
#     )
#     # The mock returns all 3, but real filtering would happen inside vector_store.
#     # We'll trust that; here we just ensure no error.
#     assert len(result["results"]) == 3


# def test_filter_by_document_id(query_service, mock_vector_store):
#     """Filter results by document_id."""
#     result = query_service.search("test", document_id="docA")
#     mock_vector_store.query.assert_called_with(
#         query_embedding=[0.1]*768,
#         n_results=5,
#         user_id=None,
#         document_id="docA",
#     )
#     assert len(result["results"]) == 3


# def test_handle_no_relevant_results(query_service, mock_vector_store):
#     """Handle queries with no relevant results."""
#     mock_vector_store.query.return_value = {
#         "ids": [],
#         "documents": [],
#         "metadatas": [],
#         "distances": [],
#     }
#     result = query_service.search("nonexistent")
#     assert len(result["results"]) == 0
#     assert result["total_results"] == 0


# def test_query_performance(query_service):
#     """Test query performance (< 500ms)."""
#     start = time.time()
#     query_service.search("performance test")
#     elapsed = time.time() - start
#     assert elapsed < 0.5  # 500ms


# def test_search_by_embedding(query_service, mock_vector_store):
#     """Test search using pre‑computed embedding."""
#     embedding = [0.2] * 768
#     result = query_service.search_by_embedding(embedding, top_k=3)
#     assert len(result["results"]) == 3
#     mock_vector_store.query.assert_called_with(
#         query_embedding=embedding,
#         n_results=3,
#         user_id=None,
#         document_id=None,
#     )


# # --- Integration Tests with FastAPI TestClient ---

# @pytest.fixture
# def client():
#     from app.main import app
#     return TestClient(app)


# def test_api_query_endpoint(client, monkeypatch):
#     """Test the full API endpoint."""
#     # Mock the query service to avoid real calls
#     mock_query_service = Mock(spec=QueryService)
#     mock_query_service.search.return_value = {
#         "results": [],
#         "total_results": 0,
#         "query_time": 0.123,
#     }
#     monkeypatch.setattr("app.api.routes.vector_store.get_query_service", lambda: mock_query_service)

#     response = client.post("/api/v1/vector-store/query", json={
#         "query_text": "test",
#         "n_results": 3,
#         "user_id": "user123",
#         "min_similarity": 0.7
#     })

#     assert response.status_code == 200
#     data = response.json()
#     assert "results" in data
#     assert data["total_results"] == 0
#     assert data["query_time"] == 0.123
#     mock_query_service.search.assert_called_once_with(
#         query_text="test",
#         top_k=3,
#         user_id="user123",
#         document_id=None,
#         min_similarity=0.7
#     )
# 

def test_api_query_endpoint():
    """Test the full API endpoint using dependency overrides."""
    from fastapi.testclient import TestClient
    from unittest.mock import Mock
    from app.main import app
    from app.api.routes import vector_store  # Add this import
    from app.services.query_service import QueryService

    # Create a mock query service
    mock_query_service = Mock(spec=QueryService)
    mock_query_service.search.return_value = {
        "results": [],
        "total_results": 0,
        "query_time": 0.123,
    }

    # Override the dependency in the app
    app.dependency_overrides[vector_store.get_query_service] = lambda: mock_query_service

    try:
        client = TestClient(app)
        response = client.post("/api/v1/vector-store/query", json={
            "query_text": "test",
            "n_results": 3,
            "user_id": "user123",
            "min_similarity": 0.7
        })

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["total_results"] == 0
        assert data["query_time"] == 0.123

        mock_query_service.search.assert_called_once_with(
            query_text="test",
            top_k=3,
            user_id="user123",
            document_id=None,
            min_similarity=0.7
        )
    finally:
        # Clean up overrides to avoid affecting other tests
        app.dependency_overrides.clear()
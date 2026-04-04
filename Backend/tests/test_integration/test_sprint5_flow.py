from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_upload_chunk_embed_store_query():
    # 1. Upload
    with open("test.txt", "rb") as f:
        upload_resp = client.post("/api/v1/documents/upload", files={"file": ("test.txt", f, "text/plain")})
    assert upload_resp.status_code == 201
    doc_id = upload_resp.json()["document_id"]

    # 2. Chunk (use the original text)
    text = "Machine learning is a subset of artificial intelligence ..."
    chunk_resp = client.post("/api/v1/chunking/chunk", json={
        "text": text,
        "document_id": doc_id,
        "chunk_size": 100,
        "chunk_overlap": 20
    })
    assert chunk_resp.status_code == 200
    chunks = chunk_resp.json()["chunks"]
    assert len(chunks) > 0

    # 3. Generate embeddings
    embed_req = {
        "texts": [chunk["text"] for chunk in chunks],
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "task_type": "retrieval_document"
    }
    embed_resp = client.post("/api/v1/embeddings/generate/batch", json=embed_req)
    assert embed_resp.status_code == 200
    embeddings = embed_resp.json()["embeddings"]
    assert all(e["success"] for e in embeddings)

    # 4. Add to vector store (requires new endpoint)
    add_req = {
        "embeddings": [e["embedding"] for e in embeddings],
        "chunks": chunks,
        "user_id": "test_user",
        "document_id": doc_id
    }
    add_resp = client.post("/api/v1/vector-store/add", json=add_req)
    assert add_resp.status_code == 200

    # 5. Query
    query_resp = client.post("/api/v1/vector-store/query", json={
        "query_text": "machine learning",
        "n_results": 3,
        "user_id": "test_user"
    })
    assert query_resp.status_code == 200
    results = query_resp.json()["results"]
    assert len(results) > 0
    assert results[0]["document"] == chunks[0]["text"]
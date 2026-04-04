"""
API routes for RAG query operations.
Complete end-to-end query processing.
"""
from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from typing import Optional, List
import time
import asyncio

from loguru import logger

from app.core.rag_engine import get_rag_engine, RAGEngine
from app.core.vector_store import get_vector_store, VectorStore
from app.core.embeddings import get_embedding_service, EmbeddingService
from app.utils.auth import get_current_user, check_rate_limit
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    CitationInfo,
    BatchQueryRequest,
    BatchQueryResponse,
    QueryStatsResponse
)

router = APIRouter()

# Query statistics tracking
_query_stats = {
    "total_queries": 0,
    "total_time": 0.0,
    "total_chunks": 0
}


@router.post(
    "/ask",
    response_model=QueryResponse,
    summary="Ask a question (RAG)",
    description="Query documents and generate answer using RAG"
)
async def ask_question(
    request: QueryRequest,
    user_id: str = Depends(get_current_user),
    rag_engine: RAGEngine = Depends(get_rag_engine),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Complete RAG query pipeline with enhanced features.
    
    Args:
        request: Query request
        user_id: Current user ID (from auth)
        rag_engine: RAG engine
        vector_store: Vector store
        
    Returns:
        Query response with answer and sources
    """
    # Check rate limit
    await check_rate_limit(user_id, "query")
    
    start_time = time.time()
    logger.info(f"Query from user {user_id}: {request.query[:100]}...")
    
    try:
        # Step 1: Generate query embedding
        retrieval_start = time.time()
        query_embedding = rag_engine.embedding_service.generate_query_embedding(request.query)
        
        # Step 2: Retrieve relevant chunks
        retrieval_results = vector_store.query(
            query_embedding=query_embedding,
            n_results=request.n_results,
            user_id=user_id,
            document_id=request.document_ids[0] if request.document_ids and len(request.document_ids) == 1 else None
        )
        
        retrieval_time = time.time() - retrieval_start
        
        # Filter by document IDs if multiple specified
        if request.document_ids and len(request.document_ids) > 1:
            filtered_results = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": []
            }
            
            for i in range(len(retrieval_results["ids"])):
                doc_id = retrieval_results["metadatas"][i].get("document_id")
                if doc_id in request.document_ids:
                    filtered_results["ids"].append(retrieval_results["ids"][i])
                    filtered_results["documents"].append(retrieval_results["documents"][i])
                    filtered_results["metadatas"].append(retrieval_results["metadatas"][i])
                    filtered_results["distances"].append(retrieval_results["distances"][i])
            
            retrieval_results = filtered_results
        
        # Filter by minimum similarity if specified
        if request.min_similarity is not None:
            filtered_results = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": []
            }
            
            for i in range(len(retrieval_results["ids"])):
                similarity = 1.0 / (1.0 + retrieval_results["distances"][i])
                if similarity >= request.min_similarity:
                    filtered_results["ids"].append(retrieval_results["ids"][i])
                    filtered_results["documents"].append(retrieval_results["documents"][i])
                    filtered_results["metadatas"].append(retrieval_results["metadatas"][i])
                    filtered_results["distances"].append(retrieval_results["distances"][i])
            
            retrieval_results = filtered_results
        
        # Format context chunks
        context_chunks = []
        sources = []
        
        for i in range(len(retrieval_results["ids"])):
            distance = retrieval_results["distances"][i]
            similarity = 1.0 / (1.0 + distance)
            metadata = retrieval_results["metadatas"][i]
            
            chunk = {
                "chunk_id": retrieval_results["ids"][i],
                "document": retrieval_results["documents"][i],
                "metadata": metadata,
                "distance": distance,
                "similarity": similarity
            }
            context_chunks.append(chunk)
            
            # Build source info
            source = SourceDocument(
                document_id=metadata.get("document_id", "unknown"),
                title=metadata.get("title", "Unknown Document"),
                chunk_id=retrieval_results["ids"][i],
                chunk_text=retrieval_results["documents"][i][:200] + "...",
                similarity=similarity,
                metadata=metadata
            )
            sources.append(source)
        
        # Step 3: Generate response
        generation_start = time.time()
        
        if context_chunks:
            response_result = rag_engine.generate_response(
                query=request.query,
                context_chunks=context_chunks,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                include_citations=request.include_citations
            )
        else:
            # No context found - generate general response
            response_result = {
                "answer": "I couldn't find any relevant information in your documents to answer this question. This might be because:\n\n1. The documents don't contain information about this topic\n2. The question requires context that isn't in your uploaded documents\n3. You haven't uploaded any documents yet\n\nPlease try:\n- Uploading relevant documents first\n- Rephrasing your question\n- Asking a different question",
                "query": request.query,
                "citations": [],
                "num_chunks_used": 0,
                "generation_time": 0,
                "success": True
            }
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # Calculate confidence score (based on similarity and citation count)
        confidence_score = None
        if context_chunks:
            avg_similarity = sum(c["similarity"] for c in context_chunks) / len(context_chunks)
            citation_factor = min(len(response_result.get("citations", [])) / max(len(context_chunks), 1), 1.0)
            confidence_score = (avg_similarity * 0.7 + citation_factor * 0.3)
        
        # Format citations
        citations = [
            CitationInfo(**cit) for cit in response_result.get("citations", [])
        ]
        
        # Update stats
        _query_stats["total_queries"] += 1
        _query_stats["total_time"] += total_time
        _query_stats["total_chunks"] += len(context_chunks)
        
        # Build response
        response = QueryResponse(
            query=request.query,
            answer=response_result["answer"],
            sources=sources,
            citations=citations,
            confidence_score=confidence_score,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            chunks_retrieved=len(context_chunks),
            chunks_used=response_result.get("num_chunks_used", 0),
            model=response_result.get("model", "unknown"),
            success=response_result.get("success", True),
            error=response_result.get("error")
        )
        
        logger.info(
            f"Query completed: {len(sources)} sources, "
            f"{total_time:.2f}s total, confidence: {confidence_score:.2f if confidence_score else 'N/A'}"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        
        return QueryResponse(
            query=request.query,
            answer=f"I encountered an error processing your query: {str(e)}",
            sources=[],
            citations=[],
            retrieval_time=0,
            generation_time=0,
            total_time=time.time() - start_time,
            chunks_retrieved=0,
            chunks_used=0,
            model="unknown",
            success=False,
            error=str(e)
        )


@router.post(
    "/batch",
    response_model=BatchQueryResponse,
    summary="Batch queries",
    description="Process multiple queries in parallel"
)
async def batch_queries(
    request: BatchQueryRequest,
    user_id: str = Depends(get_current_user),
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Process multiple queries in parallel.
    
    Args:
        request: Batch query request
        user_id: Current user ID
        rag_engine: RAG engine
        
    Returns:
        Batch query results
    """
    # Check rate limit
    await check_rate_limit(user_id, "query")
    
    logger.info(f"Batch query from user {user_id}: {len(request.queries)} queries")
    
    start_time = time.time()
    results = []
    
    # Process queries sequentially (can be made parallel with asyncio.gather)
    for query_text in request.queries:
        query_req = QueryRequest(
            query=query_text,
            n_results=request.n_results,
            temperature=request.temperature
        )
        
        try:
            # Reuse ask_question logic
            result = await ask_question(query_req, user_id, rag_engine, get_vector_store())
            results.append(result)
        except Exception as e:
            logger.error(f"Batch query failed for '{query_text}': {e}")
            results.append(QueryResponse(
                query=query_text,
                answer=f"Error: {str(e)}",
                sources=[],
                citations=[],
                retrieval_time=0,
                generation_time=0,
                total_time=0,
                chunks_retrieved=0,
                chunks_used=0,
                model="unknown",
                success=False,
                error=str(e)
            ))
    
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r.success)
    
    return BatchQueryResponse(
        results=results,
        total_queries=len(request.queries),
        successful=successful,
        failed=len(request.queries) - successful,
        total_time=total_time
    )


@router.get(
    "/stats",
    response_model=QueryStatsResponse,
    summary="Get query statistics",
    description="Get query statistics for current user"
)
async def get_query_stats(
    user_id: str = Depends(get_current_user),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Get query statistics.
    
    Args:
        user_id: Current user ID
        vector_store: Vector store
        
    Returns:
        Query statistics
    """
    # Get user's documents
    docs = vector_store.list_documents(user_id)
    total_docs = len(docs)
    total_chunks = sum(doc["chunk_count"] for doc in docs)
    
    # Calculate averages
    avg_time = (_query_stats["total_time"] / _query_stats["total_queries"] 
                if _query_stats["total_queries"] > 0 else 0)
    avg_chunks = (_query_stats["total_chunks"] / _query_stats["total_queries"] 
                  if _query_stats["total_queries"] > 0 else 0)
    
    return QueryStatsResponse(
        total_queries=_query_stats["total_queries"],
        avg_response_time=avg_time,
        avg_chunks_retrieved=avg_chunks,
        documents_indexed=total_docs,
        total_chunks=total_chunks
    )


@router.post(
    "/stream",
    summary="Stream query response",
    description="Stream answer generation (SSE)"
)
async def stream_query(
    request: QueryRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Stream query response (placeholder for streaming).
    
    Args:
        request: Query request
        user_id: Current user ID
        
    Returns:
        Streaming response
    """
    # TODO: Implement streaming with Server-Sent Events (SSE)
    # For now, return regular response
    return {"message": "Streaming not yet implemented"}


@router.get(
    "/health",
    summary="Query system health check"
)
async def query_health_check(
    rag_engine: RAGEngine = Depends(get_rag_engine),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Check query system health.
    
    Returns:
        Health status
    """
    try:
        # Test vector store
        stats = vector_store.get_collection_stats()
        
        # Test embedding service
        embedding_info = rag_engine.embedding_service.get_embedding_info()
        
        # Test LLM client
        llm_info = rag_engine.llm_client.get_model_info()
        
        return {
            "status": "healthy",
            "components": {
                "vector_store": {
                    "status": "up",
                    "total_chunks": stats["total_chunks"]
                },
                "embedding_service": {
                    "status": "up",
                    "model": embedding_info["model"]
                },
                "llm_client": {
                    "status": "up",
                    "model": llm_info["llm_model"]
                }
            },
            "query_stats": _query_stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
        
# """
#===================== Initial Query.py Code =========================================
# API routes for RAG query operations.
# """
# from fastapi import APIRouter, HTTPException, status, Depends
# from typing import Optional, List
# import time
# import asyncio

# from app.utils.auth import get_current_user

# from loguru import logger

# from app.core.rag_engine import get_rag_engine, RAGEngine
# from app.core.vector_store import get_vector_store, VectorStore
# from app.core.embeddings import get_embedding_service, EmbeddingService
# from app.utils.auth import get_current_user, check_rate_limit
# from app.models.schemas import (
#     QueryRequest,
#     QueryResponse,
#     SourceDocument,
#     CitationInfo,
#     BatchQueryRequest,
#     BatchQueryResponse,
#     QueryStatsResponse
#     # GenerateResponseRequest
# )

# router = APIRouter()

# # Query statistics tracking
# _query_stats = {
#     "total_queries":0,
#     "total_time": 0.0,
#     "total_chunks": 0
# }

# @router.post(
#     "/ask",
#     response_model=QueryResponse,
#     summary="Ask a question using RAG",
#     description="Query documents and generate an answer using retrieval-augmented generation"
# )


# async def ask_question(
#     request: QueryRequest,
#     user_id: str = Depends(get_current_user), # change from get_current_user_id to get_current_user
#     rag_engine: RAGEngine = Depends(get_rag_engine), 
#     vector_store: VectorStore = Depends(get_vector_store)):
#     """
#     Ask a question and get an answer based on your documents.
    
#     This endpoint:
#     1. Converts your question to an embedding
#     2. Retrieves relevant document chunks
#     3. Generates an answer using Gemini with the retrieved context
#     4. Returns the answer with source citations
    
#     Args:
#         request: Query request with question and filters
#         rag_engine: RAG engine instance
        
#     Returns:
#         Generated answer with citations and metadata
#     """
#     logger.info(f"RAG query request: {request.query[:100]}...")
    
#     try:
        
#         # Execute RAG pipeline
#         result = rag_engine.query(
#             query=request.query,
#             # user_id=request.user_id,
#             user_id = user_id,
#             document_id=request.document_id,
#             n_results=request.n_results,
#             temperature=request.temperature,
#             include_citations=request.include_citations
#         )
        
#         # Format citations
#         citations = [
#             CitationInfo(**citation) for citation in result.get("citations", [])
#         ]
        
#         # Create response
#         response = RAGQueryResponse(
#             answer=result["answer"],
#             query=result["query"],
#             citations=citations,
#             num_chunks_used=result.get("num_chunks_used", 0),
#             generation_time=result.get("generation_time", 0),
#             total_time=result.get("total_time", 0),
#             model=result.get("model", "unknown"),
#             success=result.get("success", True),
#             retrieval_info=result.get("retrieval_info"),
#             error=result.get("error")
#         )
        
#         if not response.success:
#             logger.warning(f"Query partially failed: {response.error}")
        
#         return response
        
#     except Exception as e:
#         logger.error(f"Query failed: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to process query: {str(e)}"
#         )


# @router.post(
#     "/generate",
#     summary="Generate response with custom context",
#     description="Generate a response using custom context (no retrieval)"
# )


# async def generate_with_context(
#     request: GenerateResponseRequest,
#     rag_engine: RAGEngine = Depends(get_rag_engine)):
#     """
#     Generate a response using provided context (bypass retrieval).
    
#     Useful for testing or when you already have the context.
    
#     Args:
#         request: Generation request with query and context
#         rag_engine: RAG engine instance
        
#     Returns:
#         Generated response
#     """
#     logger.info(f"Generate request: {request.query[:100]}...")
    
#     try:
#         # Build context chunks from provided context
#         context_chunks = [{
#             "document": request.context,
#             "metadata": {},
#             "chunk_id": "custom_context"
#         }]
        
#         # Generate response
#         result = rag_engine.generate_response(
#             query=request.query,
#             context_chunks=context_chunks,
#             system_prompt=request.system_prompt,
#             temperature=request.temperature,
#             max_tokens=request.max_tokens,
#             include_citations=request.include_citations
#         )
        
#         return {
#             "answer": result["answer"],
#             "query": result["query"],
#             "success": result["success"],
#             "generation_time": result["generation_time"],
#             "error": result.get("error")
#         }
        
#     except Exception as e:
#         logger.error(f"Generation failed: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=str(e)
#         )


# @router.get(
#     "/test-connection",
#     summary="Test RAG pipeline connection",
#     description="Test that all RAG components are working"
# )
# async def test_rag_connection(rag_engine: RAGEngine = Depends(get_rag_engine)):
#     """Test RAG pipeline components."""
#     try:
#         # Check components
#         llm_info = rag_engine.llm_client.get_model_info()
#         embedding_info = rag_engine.embedding_service.get_embedding_info()
#         vector_stats = rag_engine.vector_store.get_collection_stats()
        
#         return {
#             "status": "healthy",
#             "components": {
#                 "llm": llm_info,
#                 "embeddings": embedding_info,
#                 "vector_store": vector_stats
#             }
#         }
        
#     except Exception as e:
#         logger.error(f"Connection test failed: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=str(e)
#         )

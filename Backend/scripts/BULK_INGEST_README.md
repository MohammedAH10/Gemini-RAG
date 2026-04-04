# Bulk Document Ingestion Guide

This guide explains how to bulk ingest all documents from the `data/` directory into ChromaDB vector database.

## Overview

The bulk ingestion script processes all supported documents (PDF, EPUB, TXT, DOCX, MOBI, AZW, AZW3) from the `data/` directory and:

1. **Extracts text** from each document
2. **Chunks** the text into manageable pieces (512 tokens, 50 overlap by default)
3. **Generates embeddings** using Gemini embeddings
4. **Stores vectors** in ChromaDB with metadata for user access control

## How It Works

### System User Approach

All bulk-ingested documents are associated with a **system user ID** (`system-shared` by default). This means:

- ✅ All authenticated users can query these documents
- ✅ No duplicate data for each user
- ✅ Easy to manage shared knowledge base
- ✅ Users can still upload their own private documents

### Document Processing Pipeline

```
data/*.pdf, *.epub, *.txt, etc.
    ↓
Text Extraction (DocumentProcessor)
    ↓
Text Chunking (RAGEngine)
    ↓
Embedding Generation (Gemini API)
    ↓
Vector Storage (ChromaDB)
    ↓
Available for RAG queries
```

## Usage

### 1. Prerequisites

Make sure you have:
- ✅ Installed all dependencies: `pip install -r requirements.txt`
- ✅ Configured your `.env` file with:
  - `GEMINI_API_KEY` - Your Gemini API key
  - `SECRET_KEY` - Your application secret
  - Supabase credentials (if using authentication)

### 2. Test Run (Dry Run)

First, see what documents will be processed without actually ingesting:

```bash
python scripts/bulk_ingest.py --dry-run
```

This will list all documents found in the `data/` directory.

### 3. Full Ingestion

To ingest all documents:

```bash
python scripts/bulk_ingest.py
```

This will:
- Process all 56 documents in the `data/` directory
- Generate chunks and embeddings
- Store them in ChromaDB
- Show progress and summary

### 4. Advanced Options

#### Custom Data Directory
```bash
python scripts/bulk_ingest.py --data-dir /path/to/documents
```

#### Custom System User ID
```bash
python scripts/bulk_ingest.py --user-id your-custom-user-id
```

#### Reset Collection (⚠️ WARNING: Deletes all existing data)
```bash
python scripts/bulk_ingest.py --reset-collection
```

This is useful if you want to start fresh or rebuild the index.

## Expected Output

```
2026-04-04 10:00:00 | INFO     | Starting bulk ingestion from: /path/to/data
2026-04-04 10:00:00 | INFO     | System User ID: system-shared
2026-04-04 10:00:00 | INFO     | Found 56 documents to process
2026-04-04 10:00:00 | INFO     | Initializing indexing service...

[1/56] Processing AMS SUSTAINABLE DEVELOPMENT AND GREEN ECONOMY.pdf...
2026-04-04 10:00:15 | SUCCESS  | ✓ AMS SUSTAINABLE DEVELOPMENT AND GREEN ECONOMY.pdf: 45 chunks indexed in 12.34s

[2/56] Processing Bessemer Partners Guide to $100M ARR .pdf...
...

================================================================================
INGESTION SUMMARY
================================================================================
Total documents: 56
✓ Successful: 54
✗ Failed: 2
Total chunks indexed: 2,345
Total time: 456.78s
Average time per document: 8.16s

Failed documents:
  - Some Problem File.pdf: Error message here
================================================================================
```

## After Ingestion

### Verify Data in ChromaDB

You can check the collection statistics:

```python
from app.core.vector_store import get_vector_store

vector_store = get_vector_store()
stats = vector_store.get_collection_stats()
print(stats)
# Output: {'collection_name': 'startup_books', 'total_chunks': 2345, ...}
```

### Query the Documents

Once ingested, users can query the documents through your RAG API endpoints. The query service will filter by the system user ID to ensure all users can access the shared knowledge base.

## Troubleshooting

### "GOOGLE_API_KEY not found"
- Make sure your `.env` file is configured with a valid Gemini API key
- Get one from: https://makersuite.google.com/app/apikey

### "No module named 'xxx'"
- Run: `pip install -r requirements.txt`
- Ensure your virtual environment is activated

### Rate Limiting Errors
- Gemini API has rate limits
- The script processes documents sequentially to minimize this
- If you hit limits, wait a few minutes and re-run (it will skip already-indexed docs if you add checks)

### Memory Issues
- Large documents may consume significant memory
- Consider processing in batches if needed
- Monitor your system resources during ingestion

### ChromaDB Persistence Issues
- Ensure `data/chroma_db` directory is writable
- Check disk space (embeddings can be large)

## Integration with User Authentication

The system is designed so that:

1. **Shared Documents** (bulk ingested):
   - User ID: `system-shared`
   - Accessible to all authenticated users
   - Managed by administrators

2. **Private Documents** (user uploaded):
   - User ID: Individual user's ID
   - Only accessible to that specific user
   - Uploaded through the API

### Query Filtering

When implementing query endpoints, you can:
- Query both system-shared and user-specific documents
- Allow users to filter by document ownership
- Implement access control based on user permissions

Example query logic:
```python
# Query shared + user documents
results = vector_store.query(
    query_embedding=query_embedding,
    n_results=5,
    # Custom filter logic for system-shared OR user's docs
)
```

## Performance Tips

1. **Run during off-peak hours** - Ingestion can take 5-15 minutes for 56 docs
2. **Monitor API usage** - Check Gemini API dashboard for rate limits
3. **Use SSD storage** - ChromaDB performs better with faster I/O
4. **Consider incremental updates** - Add new documents without resetting

## Next Steps

After successful ingestion:

1. ✅ Start your FastAPI server: `python run.py`
2. ✅ Test RAG query endpoints
3. ✅ Set up user authentication (Supabase)
4. ✅ Build frontend for document management
5. ✅ Monitor usage and performance

## Support

For issues or questions:
- Check `logs/bulk_ingest.log` for detailed logs
- Review the main README.md for project setup
- See AUTH_FIXES.md for authentication details

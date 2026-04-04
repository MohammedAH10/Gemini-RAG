# Quick Start: Bulk Document Ingestion

This guide helps you quickly ingest all documents from the `data/` directory into ChromaDB.

## Prerequisites ✅

Before running the ingestion script, ensure:

1. **Dependencies installed:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment configured:**
   - Copy `.env.example` to `.env`
   - Add your `GEMINI_API_KEY`
   - Add your `SECRET_KEY` (at least 32 characters)
   - Add Supabase credentials (if using auth)

## Quick Start (3 Steps)

### Step 1: Check What Will Be Processed (Dry Run)

```bash
python scripts/bulk_ingest.py --dry-run
```

This shows all 56 documents that will be processed without actually ingesting them.

### Step 2: Run the Ingestion

```bash
python scripts/bulk_ingest.py
```

This will:
- ✅ Extract text from all PDFs, EPUBs, TXT, DOCX, MOBI files
- ✅ Chunk them into 512-token segments
- ✅ Generate embeddings using Gemini API
- ✅ Store everything in ChromaDB (`data/chroma_db/`)
- ⏱️ Takes ~5-15 minutes for 56 documents

### Step 3: Verify the Ingestion

After completion, you'll see a summary like:
```
================================================================================
INGESTION SUMMARY
================================================================================
Total documents: 56
✓ Successful: 54
✗ Failed: 2
Total chunks indexed: 2,345
Total time: 456.78s
================================================================================
```

## What Happens Next?

Once documents are ingested:

1. **All authenticated users can query these documents** via the RAG API
2. **Documents are shared** using the `system-shared` user ID
3. **Users can still upload** their own private documents
4. **ChromaDB persists** the data to `data/chroma_db/`

## Advanced Usage

### Reset and Start Fresh
If you want to clear all existing data and re-ingest:
```bash
python scripts/bulk_ingest.py --reset-collection
```
⚠️ **WARNING:** This deletes ALL existing embeddings!

### Custom User ID
```bash
python scripts/bulk_ingest.py --user-id my-custom-user-id
```

### Custom Data Directory
```bash
python scripts/bulk_ingest.py --data-dir /path/to/documents
```

## Troubleshooting

### "GOOGLE_API_KEY not found"
```bash
# Edit your .env file and add:
GEMINI_API_KEY=AIzaSy...your_actual_key
```
Get a key from: https://makersuite.google.com/app/apikey

### "Module not found"
```bash
pip install -r requirements.txt
```

### Rate Limiting Errors
- Gemini API has rate limits
- Wait a few minutes and re-run the script
- The script processes documents one-by-one to minimize this

### Out of Memory
- Large documents consume significant RAM
- Monitor your system during ingestion
- Consider processing in smaller batches

## Testing the RAG System

After ingestion, test your RAG system:

```python
from app.core.rag_engine import get_rag_engine

engine = get_rag_engine()
response = engine.query(
    query="What are the key metrics for product-market fit?",
    user_id="system-shared"  # Query shared documents
)
print(response["answer"])
```

## Document Statistics

Your `data/` directory contains:
- **PDFs:** ~35 documents
- **EPUBs:** ~8 documents  
- **Other formats:** ~13 documents

Topics covered:
- Startup fundraising & funding rounds
- Business modeling
- Product management
- Marketing & growth
- Venture capital
- Personal finance & wealth
- Project management
- Sales techniques

## Next Steps

1. ✅ Run bulk ingestion
2. ✅ Start FastAPI server: `python run.py`
3. ✅ Test RAG query endpoints
4. ✅ Set up user authentication
5. ✅ Build frontend interface

For detailed information, see:
- `scripts/BULK_INGEST_README.md` - Complete ingestion guide
- `README.md` - Project setup and architecture
- `AUTH_FIXES.md` - Authentication details

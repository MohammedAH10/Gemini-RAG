# 📚 Bulk Document Ingestion Guide

Transform your `data/` directory documents into a searchable RAG knowledge base powered by ChromaDB vector database.

## 🎯 What This Does

Processes all **56 documents** in your `data/` directory and:
- ✅ Extracts text from PDFs, EPUBs, TXT, DOCX, MOBI files
- ✅ Chunks text into semantic segments (512 tokens, 50 overlap)
- ✅ Generates embeddings using Gemini API (768 dimensions)
- ✅ Stores everything in ChromaDB with metadata
- ✅ Makes documents accessible to **all authenticated users**
- ✅ Provides detailed logging and progress tracking

## 🚀 Quick Start (3 Commands)

```bash
# 1. Verify everything is ready
python scripts/test_ingest_setup.py

# 2. See what will be processed (optional)
python scripts/bulk_ingest.py --dry-run

# 3. Run the ingestion
python scripts/bulk_ingest.py
```

**That's it!** Your documents will be indexed and ready for RAG queries.

## 📖 Alternative: Interactive Menu

For a guided experience, run:

```bash
./scripts/run_ingestion.sh
```

This provides an interactive menu with all options.

## 📊 What You'll Get

### Document Collection (56 Files)

**Startup & Fundraising:**
- Bessemer Partners Guide to $100M ARR
- Carta Startup Investment Data Cheatsheet  
- Funding Rounds, VC materials, valuation guides
- Pitch deck templates (Sequoia, Uber)

**Business Strategy:**
- The Lean Startup, Zero to One, Blitzscaling
- Business Modeling lectures (Week 1-7)
- High Growth Handbook, Startup Playbook

**Product & Sales:**
- SPIN Selling, Hooked, Contagious
- Product management guides
- Marketing templates and presentations

**Personal Development:**
- Wealth building, personal finance
- Business ethics, project management

### After Ingestion

```
Total documents: 56
Total chunks: ~2,000-3,000
Embedding dimensions: 768
Storage: ~10-20 MB in ChromaDB
Access: All authenticated users
```

## 🔧 How It Works

### Architecture

```
┌──────────────────────────────────────────────────────┐
│                   User Query                         │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│              FastAPI + RAG Engine                    │
│  1. Generate query embedding (Gemini)                │
│  2. Search ChromaDB for similar chunks               │
│  3. Retrieve top-K relevant chunks                   │
│  4. Generate answer with citations                   │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│              ChromaDB Vector Store                   │
│  user_id: "system-shared" (shared documents)         │
│  ├── Document 1 → 45 chunks                          │
│  ├── Document 2 → 32 chunks                          │
│  └── ... (56 documents)                              │
└──────────────────────────────────────────────────────┘
```

### Processing Pipeline

```
data/*.pdf, *.epub, *.txt
         ↓
  [DocumentProcessor]
  Text Extraction
         ↓
   [RAGEngine]
   Text Chunking
         ↓
  [EmbeddingService]
  Gemini Embeddings
         ↓
   [VectorStore]
   ChromaDB Storage
         ↓
  Ready for Queries!
```

## 💡 Design Decisions

### System User Approach

All bulk-ingested documents use `user_id: "system-shared"`:

**Benefits:**
- ✅ All authenticated users can query these documents
- ✅ No data duplication across users
- ✅ Easy to manage shared knowledge base
- ✅ Users can still upload private documents

**Query Example:**
```python
from app.core.rag_engine import get_rag_engine

engine = get_rag_engine()

# Query shared documents
response = engine.query(
    query="What are key startup metrics?",
    user_id="system-shared"
)

print(response["answer"])
```

## 🛠️ Advanced Usage

### Custom User ID
```bash
python scripts/bulk_ingest.py --user-id my-custom-id
```

### Custom Data Directory
```bash
python scripts/bulk_ingest.py --data-dir /path/to/documents
```

### Reset and Rebuild
```bash
python scripts/bulk_ingest.py --reset-collection
```
⚠️ **WARNING:** Deletes all existing data!

### Check Status
```bash
python scripts/check_chroma_status.py
```

## 📋 Prerequisites

Before running ingestion, ensure:

### 1. Dependencies Installed
```bash
pip install -r requirements.txt
```

### 2. Environment Configured
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add:
GEMINI_API_KEY=AIzaSy...your_key
SECRET_KEY=your_secret_at_least_32_chars
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_key
```

### 3. Gemini API Key
Get one from: https://makersuite.google.com/app/apikey

## ✅ Verification

### Test Setup
```bash
python scripts/test_ingest_setup.py
```

**Expected Output:**
```
✓ PASS - Imports
✓ PASS - Configuration
✓ PASS - Document Scan
✓ PASS - Vector Store
✓ PASS - Gemini API

✅ All tests passed! You're ready to run bulk ingestion.
```

### Check ChromaDB
```bash
python scripts/check_chroma_status.py
```

**Expected Output:**
```
✓ Collection Name: startup_books
✓ Total Chunks: 2345
✓ Has Data: True
```

## ⏱️ Performance

### Time Estimates
| Collection Size | Estimated Time |
|----------------|----------------|
| 10 documents   | 1-2 minutes    |
| 56 documents   | 5-15 minutes   |
| 100+ documents | 15-30 minutes  |

### Resource Usage
- **Memory:** 100-500 MB (temporary spikes)
- **Disk:** ~10-20 MB for ChromaDB
- **API Calls:** ~50-100 Gemini requests total

## 🔍 Troubleshooting

### "GOOGLE_API_KEY not found"
```bash
# Edit .env file
GEMINI_API_KEY=AIzaSy...your_actual_key
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### Rate Limiting Errors
- Gemini API has rate limits (60 req/min typically)
- Wait 5-10 minutes
- Re-run the script

### Out of Memory
- Monitor RAM during ingestion
- Large documents can spike to 500 MB temporarily
- Normal after processing completes

### ChromaDB Issues
```bash
# Ensure directory is writable
chmod -R 755 data/chroma_db
```

## 📚 Your Documents

Complete list of what will be processed:

**PDFs (35+):**
- AMS SUSTAINABLE DEVELOPMENT AND GREEN ECONOMY.pdf
- Bessemer Partners Guide to $100M ARR .pdf
- Blitzscaling. The Lightning Fast Path.pdf
- Building Skills for a Future-Ready Career in Tech.pdf
- Business Modeling Week 1-7 Lecture Documents
- CARTA Startup investment data cheatsheet 2024 2025.pdf
- High Growth Handbook.pdf
- Nigeria Startup Act 2022.pdf
- Zero to One (Peter Thiel).pdf
- And 25+ more...

**EPUBs (8+):**
- Where startup ideas come from.epub
- The Rules of Wealth.epub
- How Big Things Get Done.epub
- The $100 Startup.epub
- The Millionaire Fastlane.epub
- And 3+ more...

**Other Formats:**
- Start-up Marketing Presentation.pptx
- UX DESIGN FOR START UPS.pptx
- Various templates and guides

## 🎯 Next Steps After Ingestion

1. ✅ **Verify:** Check chunk counts with `check_chroma_status.py`
2. ✅ **Test Query:** Use RAG engine to query documents
3. ✅ **Start Server:** `python run.py`
4. ✅ **Test API:** Hit your query endpoints
5. ✅ **Setup Auth:** Configure Supabase for users
6. ✅ **Build Frontend:** Create UI for users
7. ✅ **Monitor:** Track API usage and performance

## 📁 File Structure

```
scripts/
├── bulk_ingest.py              # Main ingestion script
├── check_chroma_status.py      # Verify ChromaDB state
├── test_ingest_setup.py        # Test prerequisites
├── run_ingestion.sh            # Interactive menu
├── QUICK_START.md             # Quick start guide
├── BULK_INGEST_README.md      # Detailed documentation
└── BULK_INGEST_SUMMARY.md     # Implementation summary

data/
├── [56 documents]              # Your source documents
└── chroma_db/                  # ChromaDB storage (after ingestion)

logs/
└── bulk_ingest.log            # Detailed ingestion logs
```

## 🔗 Additional Resources

- **Quick Start:** `scripts/QUICK_START.md`
- **Full Guide:** `scripts/BULK_INGEST_README.md`
- **Implementation:** `scripts/BULK_INGEST_SUMMARY.md`
- **Project README:** `README.md`
- **Auth Setup:** `AUTH_FIXES.md`
- **ChromaDB Docs:** https://docs.trychroma.com/
- **Gemini API:** https://ai.google.dev/docs

## 💬 Example Queries

After ingestion, you can query:

```python
# Startup metrics
"What are the key metrics for product-market fit?"

# Fundraising
"How do I prepare for Series A funding?"

# Business strategy
"What is the lean startup methodology?"

# Product development
"How to achieve product-market fit?"

# Marketing
"What are effective growth hacking strategies?"

# Pitch decks
"What should be in a seed round pitch deck?"
```

## 🎉 Success Criteria

After running ingestion, you should have:

- ✅ 2,000-3,000 chunks in ChromaDB
- ✅ All chunks tagged with `user_id="system-shared"`
- ✅ Ability to query all 56 documents
- ✅ Persistent storage in `data/chroma_db/`
- ✅ Detailed logs in `logs/bulk_ingest.log`
- ✅ Ready for production RAG queries

## 🆘 Need Help?

1. Check logs: `tail -f logs/bulk_ingest.log`
2. Verify setup: `python scripts/test_ingest_setup.py`
3. Check status: `python scripts/check_chroma_status.py`
4. Review docs: See README files in `scripts/` directory

---

**Ready to build your startup knowledge base? 🚀**

```bash
python scripts/bulk_ingest.py
```

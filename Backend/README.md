# RAG Startup Books - Backend

A production-ready RAG (Retrieval-Augmented Generation) system for querying PDF documents using Gemini, LlamaIndex, ChromaDB, and FastAPI with Supabase authentication.

## Authentication Setup

This project supports two authentication methods:
1. **Email/Password Authentication** - Traditional signup/signin
2. **Google OAuth** - OAuth-based authentication for web and mobile apps

### Supabase Authentication Configuration

#### 1. Create a Supabase Project
1. Go to [Supabase](https://supabase.com) and create a new project
2. Get your project URL and keys from Settings > API

#### 2. Configure Supabase Auth
1. Go to Authentication > Providers in your Supabase dashboard
2. For **Email/Password**: It's enabled by default
3. For **Google OAuth**:
   - Go to Authentication > Providers > Google
   - Enable the provider
   - Get your Google OAuth credentials from [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
   - Create OAuth 2.0 Client ID credentials
   - Add your Client ID and Client Secret to Supabase
   - Add authorized redirect URIs: `https://your-project.supabase.co/auth/v1/callback`

#### 3. Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Supabase credentials
```

Required Supabase variables:
```bash
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_KEY="your-supabase-anon-key"
SUPABASE_SERVICE_ROLE_KEY="your-supabase-service-role-key"
OAUTH_REDIRECT_URL="http://localhost:8000/api/v1/auth/google/callback"
OAUTH_GOOGLE_CLIENT_ID="your-google-client-id.apps.googleusercontent.com"
```

#### 4. Configure Google Cloud Console
1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create OAuth 2.0 Client ID (Web application type)
3. Add authorized redirect URIs:
   - `https://your-project.supabase.co/auth/v1/callback`
   - `http://localhost:8000/api/v1/auth/google/callback` (for development)
4. Add authorized JavaScript origins:
   - `http://localhost:8000` (for development)
   - Your production domain

## Auth API Endpoints

### Email/Password Authentication

#### Sign Up
```bash
POST /api/v1/auth/signup
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

#### Sign In
```bash
POST /api/v1/auth/signin
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

#### Sign Out
```bash
POST /api/v1/auth/signout
Authorization: Bearer <access_token>
```

### Google OAuth Authentication

#### Get OAuth URL (Web Flow)
```bash
GET /api/v1/auth/google?redirect_to=http://localhost:3000/callback
```
This redirects to Google's OAuth page.

#### OAuth Callback
```bash
GET /api/v1/auth/google/callback?code=<authorization_code>&redirect_url=http://localhost:8000/api/v1/auth/google/callback
```

#### Mobile/SPA Authentication
```bash
POST /api/v1/auth/google/mobile?id_token=<google_id_token>
```

### Token Management

#### Refresh Token
```bash
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "your_refresh_token"
}
```

#### Get Current User
```bash
GET /api/v1/auth/me
Authorization: Bearer <access_token>
```

#### Verify Token
```bash
GET /api/v1/auth/verify
Authorization: Bearer <access_token>
```

#### Reset Password
```bash
POST /api/v1/auth/reset-password
Content-Type: application/json

{
  "email": "user@example.com"
}
```

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   └── auth.py              # Authentication endpoints
│   │   └── dependencies.py          # FastAPI dependencies
│   ├── core/                        # Core RAG components
│   │   ├── document_processor.py    # PDF text extraction
│   │   ├── embeddings.py            # Gemini embeddings
│   │   └── vector_store.py          # ChromaDB integration
│   ├── models/
│   │   ├── schemas.py               # Pydantic models
│   │   └── database.py              # Database models
│   ├── services/
│   │   ├── supabase_service.py      # Supabase auth & DB operations
│   │   └── indexing_service.py      # Document indexing pipeline
│   ├── utils/
│   │   └── auth.py                  # Auth middleware & utilities
│   ├── config.py                    # Configuration management
│   └── main.py                      # FastAPI app entry point
├── data/
│   ├── pdfs/                        # PDF storage
│   └── chroma_db/                   # Vector database persistence
├── tests/
│   └── test_sprint8.py              # Auth & rate limiting tests
├── test/
│   └── test_integration_e2e.py      # End-to-end tests
├── conftest.py                      # Pytest configuration
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
└── README.md                        # This file
```

## Batch 1: Core RAG Pipeline

This batch includes:
  - PDF text extraction (PyPDF2 + pdfplumber)
  - Text chunking with LlamaIndex
  - Gemini embeddings generation
  - ChromaDB vector storage
  - Complete indexing pipeline

## Setup Instructions

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Gemini API key
# Get your API key from: https://makersuite.google.com/app/apikey
```

Example `.env` file:
```
GOOGLE_API_KEY=your_actual_gemini_api_key_here
CHROMA_PERSIST_DIR=./data/chroma_db
PDF_STORAGE_PATH=./data/pdfs
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

### 3. Run Batch 1 Tests

```bash
# Run all Batch 1 tests
python test_batch1.py
```

The test script will:
1. Create a sample PDF document
2. Extract text from the PDF
3. Generate embeddings using Gemini
4. Store vectors in ChromaDB
5. Test retrieval with sample queries

## Expected Test Output

```
🚀 STARTING BATCH 1 TESTS

============================================================
TEST 1: Document Processing
============================================================
✓ Extracted text from 3 pages
Page 1 preview: Startup Fundraising Guide...

============================================================
TEST 2: Embeddings Generation
============================================================
✓ Generated embedding with 768 dimensions
✓ Generated 3 batch embeddings

============================================================
TEST 3: Vector Store (ChromaDB)
============================================================
✓ Created collection: test_startup_docs
✓ Added 3 documents to vector store
✓ Retrieved 2 results

============================================================
TEST 4: Complete Indexing Pipeline
============================================================
✓ Indexing Results:
  document_id: startup_guide_001
  total_pages: 3
  total_chunks: 15
  status: success

============================================================
TEST 5: End-to-End Query Test
============================================================
Query: 'What are the key metrics for product-market fit?'
Found 3 relevant chunks

✅ ALL BATCH 1 TESTS PASSED!
```

## Component Details

### DocumentProcessor
Extracts text from PDFs with automatic fallback:
- Primary: PyPDF2 (fast, good for most PDFs)
- Fallback: pdfplumber (better for complex layouts)
- Automatic text cleaning and normalization

### GeminiEmbeddings
Generates 768-dimensional embeddings using Google's Gemini API:
- Optimized for document indexing (RETRIEVAL_DOCUMENT)
- Optimized for queries (RETRIEVAL_QUERY)
- Batch processing with rate limit handling

### ChromaVectorStore
Persistent vector database for embeddings:
- Automatic persistence to disk
- Metadata filtering support
- Similarity search with distance metrics

### IndexingService
Orchestrates the complete pipeline:
1. PDF text extraction
2. Text chunking (512 tokens, 50 overlap)
3. Embedding generation
4. Vector storage with metadata

## Troubleshooting

### "GOOGLE_API_KEY not found"
Make sure you've created a `.env` file and added your Gemini API key.

### "No module named 'reportlab'"
The test script uses reportlab to create sample PDFs. Install it:
```bash
pip install reportlab
```

### ChromaDB persistence issues
Make sure the `data/chroma_db` directory is writable.

### Rate limiting errors
Gemini API has rate limits. The code includes batch processing, but if you hit limits:
- Reduce batch size in embeddings.py
- Add delays between requests

## Next Steps (Batch 2)

After Batch 1 passes, we'll build:
- Query engine with context preparation
- LLM integration for answer generation
- Response post-processing
- Complete RAG query pipeline

## Configuration Options

All settings in `.env`:

```bash
# Gemini
GOOGLE_API_KEY=your_key
EMBEDDING_MODEL=models/embedding-001
LLM_MODEL=gemini-pro

# Chunking
CHUNK_SIZE=512          # Token size per chunk
CHUNK_OVERLAP=50        # Overlap between chunks

# Storage
CHROMA_PERSIST_DIR=./data/chroma_db
PDF_STORAGE_PATH=./data/pdfs
MAX_FILE_SIZE_MB=25
```

## API Key Setup

1. Go to https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key
4. Add to `.env` file

## License

MIT License
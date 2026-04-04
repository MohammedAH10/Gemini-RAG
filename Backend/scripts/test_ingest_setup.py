"""
Test script to verify bulk ingestion setup is working correctly.
Runs basic checks without processing all documents.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from app.config import get_settings
        print("✓ app.config")
        
        from app.services.indexing_service import IndexingService
        print("✓ app.services.indexing_service")
        
        from app.services.document_processor import DocumentProcessor
        print("✓ app.services.document_processor")
        
        from app.core.vector_store import get_vector_store
        print("✓ app.core.vector_store")
        
        from app.core.rag_engine import get_rag_engine
        print("✓ app.core.rag_engine")
        
        from app.core.embeddings import get_embedding_service
        print("✓ app.core.embeddings")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test that configuration loads correctly."""
    print("\nTesting configuration...")
    
    try:
        from app.config import get_settings
        settings = get_settings()
        
        print(f"✓ App Name: {settings.app_name}")
        print(f"✓ Environment: {settings.environment}")
        print(f"✓ Chunk Size: {settings.chunk_size}")
        print(f"✓ Chunk Overlap: {settings.chunk_overlap}")
        print(f"✓ Collection Name: {settings.chroma_collection_name}")
        print(f"✓ Allowed Extensions: {settings.allowed_extensions}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False


def test_document_scan():
    """Test that documents can be found in data directory."""
    print("\nTesting document scan...")
    
    try:
        from scripts.bulk_ingest import get_document_files
        
        data_dir = project_root / "data"
        doc_files = get_document_files(data_dir)
        
        print(f"✓ Found {len(doc_files)} documents")
        
        if doc_files:
            print(f"\nFirst 5 documents:")
            for i, file_path in enumerate(doc_files[:5], 1):
                print(f"  {i}. {file_path.name}")
        
        return True
    except Exception as e:
        print(f"✗ Document scan failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_store():
    """Test that vector store initializes correctly."""
    print("\nTesting vector store...")
    
    try:
        from app.core.vector_store import get_vector_store
        
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        
        print(f"✓ Vector store initialized")
        print(f"✓ Collection: {stats['collection_name']}")
        print(f"✓ Total chunks: {stats['total_chunks']}")
        print(f"✓ Has data: {stats['has_data']}")
        
        return True
    except Exception as e:
        print(f"✗ Vector store failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gemini_api():
    """Test that Gemini API key is configured."""
    print("\nTesting Gemini API configuration...")
    
    try:
        from app.config import get_settings
        settings = get_settings()
        
        if settings.gemini_api_key and settings.gemini_api_key != "AIzaSyYourGeminiApiKeyHere":
            print(f"✓ Gemini API key configured (starts with: {settings.gemini_api_key[:8]}...)")
            return True
        else:
            print("✗ Gemini API key not properly configured")
            print("  Please set GEMINI_API_KEY in your .env file")
            return False
    except Exception as e:
        print(f"✗ Gemini API test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("BULK INGESTION SETUP VERIFICATION")
    print("=" * 80)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Document Scan", test_document_scan),
        ("Vector Store", test_vector_store),
        ("Gemini API", test_gemini_api),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed! You're ready to run bulk ingestion.")
        print("\nNext steps:")
        print("  1. Run dry-run: python scripts/bulk_ingest.py --dry-run")
        print("  2. Run ingestion: python scripts/bulk_ingest.py")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues before running ingestion.")
        print("\nCommon issues:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - Missing .env file: cp .env.example .env")
        print("  - Invalid GEMINI_API_KEY: Get one from https://makersuite.google.com/app/apikey")
        return 1


if __name__ == "__main__":
    sys.exit(main())

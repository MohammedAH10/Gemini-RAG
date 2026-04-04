"""
Bulk document ingestion script.
Processes all documents from the data/ directory and indexes them into ChromaDB.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from typing import Optional
from uuid import uuid4

from loguru import logger
from app.config import get_settings
from app.services.indexing_service import IndexingService
from app.services.document_processor import DocumentProcessor

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <message>")
logger.add("logs/bulk_ingest.log", rotation="10 MB", level="DEBUG")

# System user ID for shared documents
# All authenticated users will be able to access these
SYSTEM_USER_ID = "system-shared"


def get_document_files(data_dir: Path) -> list[Path]:
    """
    Get all supported document files from the data directory.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        List of document file paths
    """
    settings = get_settings()
    allowed_extensions = settings.allowed_extensions_list
    
    # Get all files with supported extensions
    doc_files = []
    for ext in allowed_extensions:
        doc_files.extend(data_dir.glob(f"*.{ext}"))
    
    # Remove duplicates and sort
    doc_files = sorted(list(set(doc_files)))
    
    return doc_files


def process_single_document(
    indexing_service: IndexingService,
    file_path: Path,
    system_user_id: str,
    document_id: Optional[str] = None
) -> dict:
    """
    Process and index a single document.
    
    Args:
        indexing_service: IndexingService instance
        file_path: Path to document file
        system_user_id: System user ID for shared access
        document_id: Optional document ID (generated if not provided)
        
    Returns:
        Processing result dictionary
    """
    start_time = time.time()
    
    if not document_id:
        # Generate document ID from filename
        document_id = file_path.stem.replace(" ", "_").lower()
        # Ensure it's unique by adding timestamp
        document_id = f"{document_id}_{int(time.time())}"
    
    logger.info(f"Processing: {file_path.name} (ID: {document_id})")
    
    try:
        # Use the indexing service which handles the full pipeline:
        # 1. Text extraction
        # 2. Chunking
        # 3. Embedding generation
        # 4. Vector storage
        
        result = indexing_service.index_document(
            document_id=document_id,
            user_id=system_user_id,
            file_path=str(file_path),
            title=file_path.stem
        )
        
        processing_time = time.time() - start_time
        
        if result.get("success"):
            logger.success(
                f"✓ {file_path.name}: "
                f"{result['chunk_count']} chunks indexed in {processing_time:.2f}s"
            )
            return {
                "success": True,
                "filename": file_path.name,
                "document_id": document_id,
                "chunks": result.get("chunk_count", 0),
                "time": processing_time
            }
        else:
            logger.error(
                f"✗ {file_path.name}: {result.get('error', 'Unknown error')}"
            )
            return {
                "success": False,
                "filename": file_path.name,
                "document_id": document_id,
                "error": result.get("error"),
                "time": processing_time
            }
            
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"✗ {file_path.name}: {str(e)}")
        return {
            "success": False,
            "filename": file_path.name,
            "document_id": document_id,
            "error": str(e),
            "time": processing_time
        }


def bulk_ingest_documents(
    data_dir: Optional[Path] = None,
    system_user_id: str = SYSTEM_USER_ID,
    dry_run: bool = False
) -> dict:
    """
    Bulk ingest all documents from the data directory.
    
    Args:
        data_dir: Path to data directory (defaults to ./data)
        system_user_id: System user ID for shared access
        dry_run: If True, only scan and report without processing
        
    Returns:
        Ingestion summary dictionary
    """
    if data_dir is None:
        data_dir = project_root / "data"
    
    logger.info(f"Starting bulk ingestion from: {data_dir}")
    logger.info(f"System User ID: {system_user_id}")
    logger.info(f"Dry Run: {dry_run}")
    
    # Get document files
    doc_files = get_document_files(data_dir)
    
    if not doc_files:
        logger.warning("No documents found in the data directory!")
        return {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "results": []
        }
    
    logger.info(f"Found {len(doc_files)} documents to process")
    logger.info("=" * 80)
    
    if dry_run:
        logger.info("DRY RUN - Listing documents without processing:")
        for i, file_path in enumerate(doc_files, 1):
            logger.info(f"  {i}. {file_path.name}")
        return {
            "total": len(doc_files),
            "successful": 0,
            "failed": 0,
            "dry_run": True,
            "results": [{"filename": f.name} for f in doc_files]
        }
    
    # Initialize services
    logger.info("Initializing indexing service...")
    indexing_service = IndexingService()
    
    # Process each document
    results = []
    successful = 0
    failed = 0
    total_chunks = 0
    total_start = time.time()
    
    for i, file_path in enumerate(doc_files, 1):
        logger.info(f"\n[{i}/{len(doc_files)}] Processing {file_path.name}...")
        
        result = process_single_document(
            indexing_service=indexing_service,
            file_path=file_path,
            system_user_id=system_user_id
        )
        
        results.append(result)
        
        if result["success"]:
            successful += 1
            total_chunks += result.get("chunks", 0)
        else:
            failed += 1
    
    total_time = time.time() - total_start
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total documents: {len(doc_files)}")
    logger.success(f"✓ Successful: {successful}")
    if failed > 0:
        logger.error(f"✗ Failed: {failed}")
    logger.info(f"Total chunks indexed: {total_chunks}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average time per document: {total_time / len(doc_files):.2f}s")
    
    # Show failed documents
    if failed > 0:
        logger.warning("\nFailed documents:")
        for result in results:
            if not result["success"]:
                logger.warning(f"  - {result['filename']}: {result.get('error')}")
    
    logger.info("=" * 80)
    
    return {
        "total": len(doc_files),
        "successful": successful,
        "failed": failed,
        "total_chunks": total_chunks,
        "total_time": total_time,
        "results": results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bulk ingest documents into ChromaDB")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (default: ./data)"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=SYSTEM_USER_ID,
        help=f"System user ID for shared access (default: {SYSTEM_USER_ID})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan and list documents without processing"
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Reset the ChromaDB collection before ingestion (WARNING: deletes all data!)"
    )
    
    args = parser.parse_args()
    
    try:
        # Optional: Reset collection if requested
        if args.reset_collection:
            from app.core.vector_store import get_vector_store
            logger.warning("Resetting ChromaDB collection...")
            vector_store = get_vector_store()
            vector_store.reset_collection()
            logger.success("Collection reset complete")
        
        # Run bulk ingestion
        data_dir = Path(args.data_dir) if args.data_dir else None
        summary = bulk_ingest_documents(
            data_dir=data_dir,
            system_user_id=args.user_id,
            dry_run=args.dry_run
        )
        
        # Exit with appropriate code
        if summary["failed"] > 0:
            sys.exit(1)
        elif summary["total"] == 0:
            sys.exit(0)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.warning("\nIngestion cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Bulk ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

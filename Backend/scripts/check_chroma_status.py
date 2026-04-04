"""
ChromaDB verification and statistics script.
Check the current state of documents in the vector database.
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from app.core.vector_store import get_vector_store
from app.config import get_settings

def check_chroma_status():
    """Check and display ChromaDB status."""
    
    print("=" * 80)
    print("CHROMADB VECTOR DATABASE STATUS")
    print("=" * 80)
    
    try:
        # Get vector store
        vector_store = get_vector_store()
        
        # Get collection stats
        stats = vector_store.get_collection_stats()
        
        print(f"\n✓ Collection Name: {stats['collection_name']}")
        print(f"✓ Total Chunks: {stats['total_chunks']}")
        print(f"✓ Persist Directory: {stats['persist_directory']}")
        print(f"✓ Has Data: {stats['has_data']}")
        
        if stats['total_chunks'] > 0:
            # Get sample data to show structure
            print("\n" + "=" * 80)
            print("SAMPLE CHUNK DATA")
            print("=" * 80)
            
            # Get a few chunks to display
            sample = vector_store.collection.peek(limit=3)
            
            if sample and sample['ids']:
                for i, (chunk_id, doc, metadata) in enumerate(
                    zip(sample['ids'], sample['documents'], sample['metadatas'])
                ):
                    print(f"\n--- Chunk {i+1} ---")
                    print(f"ID: {chunk_id}")
                    print(f"Text (first 100 chars): {doc[:100]}...")
                    print(f"Metadata: {metadata}")
            
            # Group by document_id to show document distribution
            print("\n" + "=" * 80)
            print("DOCUMENT DISTRIBUTION")
            print("=" * 80)
            
            # Get all metadata to group by document
            all_data = vector_store.collection.get(
                include=["metadatas"],
                limit=min(stats['total_chunks'], 1000)  # Limit for performance
            )
            
            if all_data['metadatas']:
                doc_counts = defaultdict(int)
                user_counts = defaultdict(int)
                
                for meta in all_data['metadatas']:
                    doc_id = meta.get('document_id', 'unknown')
                    user_id = meta.get('user_id', 'unknown')
                    doc_counts[doc_id] += 1
                    user_counts[user_id] += 1
                
                print(f"\nUnique Documents (sample): {len(doc_counts)}")
                print(f"Unique Users: {len(user_counts)}")
                
                print("\nUser Distribution:")
                for user_id, count in sorted(user_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {user_id}: {count} chunks")
                
                print("\nTop 10 Documents by Chunk Count:")
                for doc_id, count in sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {doc_id}: {count} chunks")
        
        print("\n" + "=" * 80)
        print("CONFIGURATION")
        print("=" * 80)
        
        settings = get_settings()
        print(f"✓ Chunk Size: {settings.chunk_size}")
        print(f"✓ Chunk Overlap: {settings.chunk_overlap}")
        print(f"✓ Embedding Model: {settings.gemini_embedding_model}")
        print(f"✓ Collection Name: {settings.chroma_collection_name}")
        print(f"✓ Persist Directory: {settings.chroma_persist_directory}")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error checking ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    check_chroma_status()

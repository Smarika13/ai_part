"""
Script to ingest JSON data and create vector store
Run this once to build the initial index, or when you update your data
"""

print("=" * 50)
print("DEBUG: Script is starting...")
print("=" * 50)

import sys
from pathlib import Path

print("DEBUG: Basic imports successful")

# Add parent directory to Python path so we can import from app
sys.path.append(str(Path(__file__).parent.parent))

print("DEBUG: Path added, about to import RAGService")

try:
    from app.services.rag_service import RAGService
    print("DEBUG: RAGService imported successfully!")
except Exception as e:
    print(f"ERROR importing RAGService: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def main():
    print("\n" + "=" * 50)
    print("Chitwan National Park RAG - Data Ingestion")
    print("=" * 50)
    
    try:
        print("DEBUG: Creating RAGService instance...")
        rag = RAGService()
        
        print("DEBUG: Initializing RAG with rebuild_index=True...")
        rag.initialize(rebuild_index=True)
        
        print("\n" + "=" * 50)
        print("✓ Data ingestion completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Error during data ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("DEBUG: About to call main()")

if __name__ == "__main__":
    main()
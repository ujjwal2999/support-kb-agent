"""
CLI entrypoint

Usage:
    python -m src.main ingest
    python -m src.main query "your question here"
"""
import sys

from src.config import DATA_DIR
from src.loader import load_documents, chunk_documents
from src.store import store
from src.graph import query as run_query


def ingest_command():
    """load and index documents"""
    print(f"\n📚 Loading documents from {DATA_DIR}...")
    
    docs = load_documents(DATA_DIR)
    if not docs:
        print("❌ No documents found!")
        return
    
    print(f"\n✂️  Chunking {len(docs)} documents...")
    chunks = chunk_documents(docs)
    
    print(f"\n🔢 Building vector index...")
    store.add_documents(chunks)
    store.save()
    
    print(f"\n✅ Done! Indexed {len(docs)} documents into {len(chunks)} chunks.")


def query_command(question: str):
    """answer a question from the KB"""
    if not store.load():
        print("❌ No index found. Run 'python -m src.main ingest' first.")
        return
    
    print(f"\n🔍 Query: {question}")
    print("=" * 60)
    
    answer = run_query(question)
    print(answer)
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    cmd = sys.argv[1]
    
    if cmd == "ingest":
        ingest_command()
    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Usage: python -m src.main query <question>")
            return
        question = " ".join(sys.argv[2:])
        query_command(question)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()

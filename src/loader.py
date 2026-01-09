"""
Loads and chunks documents from the data folder
"""
from pathlib import Path
from typing import List, Dict
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR


def load_documents(directory: Path = DATA_DIR) -> List[Dict]:
    """read all .md and .txt files"""
    documents = []
    
    for ext in ['.md', '.txt']:
        for filepath in directory.glob(f"*{ext}"):
            content = filepath.read_text(encoding='utf-8')
            documents.append({
                "content": content,
                "source": filepath.name
            })
            print(f"  Loaded: {filepath.name}")
    
    return documents


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Split docs into overlapping chunks using sliding window.
    Tried recursive splitting first but this is simpler and works fine.
    """
    chunks = []
    
    for doc in documents:
        text = doc["content"]
        source = doc["source"]
        
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "source": source,
                    "chunk_id": chunk_idx
                })
            
            if end >= len(text):
                break
            
            # move forward with overlap
            start = end - CHUNK_OVERLAP
            if start <= 0:
                start = end
            chunk_idx += 1
    
    return chunks

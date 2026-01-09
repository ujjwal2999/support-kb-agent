"""
Vector store with FAISS + TF-IDF

I wanted to use Gemini embeddings but kept getting rate limited so 
switched to TF-IDF. Works well enough for this use case tbh.
"""
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

from src.config import INDEX_PATH, DEFAULT_K


class VectorStore:
    """FAISS vector store with TF-IDF embeddings"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=384,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.index: Optional[faiss.IndexFlatL2] = None
        self.documents: List[Dict] = []
    
    def add_documents(self, docs: List[Dict]) -> None:
        """add documents and build the index"""
        texts = [doc["content"] for doc in docs]
        
        vectors = self.vectorizer.fit_transform(texts).toarray()
        vectors = vectors.astype('float32')
        
        # normalize vectors for cosine sim
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)
        
        self.documents = docs
        print(f"  Indexed {len(docs)} chunks")
    
    def search(self, query: str, k: int = DEFAULT_K) -> List[Dict]:
        """search for similar docs"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_vec = self.vectorizer.transform([query]).toarray()
        query_vec = query_vec.astype('float32')
        
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(distances[0][i])
                results.append(doc)
        
        return results
    
    def save(self) -> None:
        """save index to disk"""
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(INDEX_PATH / "vectors.faiss"))
        
        with open(INDEX_PATH / "metadata.pkl", "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "documents": self.documents
            }, f)
        
        print(f"  Saved to {INDEX_PATH}")
    
    def load(self) -> bool:
        """load index from disk"""
        index_file = INDEX_PATH / "vectors.faiss"
        meta_file = INDEX_PATH / "metadata.pkl"
        
        if not index_file.exists() or not meta_file.exists():
            return False
        
        self.index = faiss.read_index(str(index_file))
        
        with open(meta_file, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.documents = data["documents"]
        
        print(f"  Loaded {len(self.documents)} chunks from index")
        return True


# global instance for easy access
store = VectorStore()

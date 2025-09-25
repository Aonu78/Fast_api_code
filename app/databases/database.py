import faiss
import numpy as np
from typing import List, Dict
import os
import pickle

class VectorStore:
    def __init__(self):
        self.dimension = 1536  # OpenAI ada-002 embedding dimension
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        self.documents: Dict[str, Dict] = {}
        self.chunk_metadata: List[Dict] = []
        self.load_index()

    def load_index(self):
        if os.path.exists("faiss_index.bin") and os.path.exists("metadata.pkl"):
            self.index = faiss.read_index("faiss_index.bin")
            with open("metadata.pkl", "rb") as f:
                self.documents, self.chunk_metadata = pickle.load(f)

    def save_index(self):
        faiss.write_index(self.index, "faiss_index.bin")
        with open("metadata.pkl", "wb") as f:
            pickle.dump((self.documents, self.chunk_metadata), f)

    def add_document(self, doc_id: str, title: str, chunks: List[str], embeddings: List[List[float]]):
        self.documents[doc_id] = {"title": title, "content": "\n".join(chunks)}

        start_index = len(self.chunk_metadata)
        vector_ids = list(range(start_index, start_index + len(chunks)))

        for i, chunk in enumerate(chunks):
            self.chunk_metadata.append({
                "doc_id": doc_id,
                "title": title,
                "chunk": chunk,
                "embedding": embeddings[i]
            })

        embeddings_np = np.array(embeddings, dtype=np.float32)
        vector_ids_np = np.array(vector_ids, dtype=np.int64)

        self.index.add_with_ids(embeddings_np, vector_ids_np)
        self.save_index()

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        distances, vector_ids = self.index.search(query_embedding_np, k)

        results = []
        for i, vector_id in enumerate(vector_ids[0]):
            if vector_id != -1 and vector_id < len(self.chunk_metadata):
                metadata = self.chunk_metadata[vector_id]
                results.append({
                    "id": metadata["doc_id"],
                    "title": metadata["title"],
                    "chunk": metadata["chunk"]
                })
        return results

    def get_document(self, doc_id: str) -> Dict:
        doc = self.documents.get(doc_id)
        if doc:
            return {
                "id": doc_id,
                "title": doc["title"],
                "content": doc["content"]
            }
        return None

    def delete_document(self, doc_id: str):
        if doc_id not in self.documents:
            return False

        del self.documents[doc_id]

        self.chunk_metadata = [meta for meta in self.chunk_metadata if meta['doc_id'] != doc_id]

        self.index.reset()

        if self.chunk_metadata:
            embeddings = np.array([meta["embedding"] for meta in self.chunk_metadata], dtype=np.float32)
            vector_ids = np.arange(len(self.chunk_metadata), dtype=np.int64)
            self.index.add_with_ids(embeddings, vector_ids)

        self.save_index()
        return True

    def list_documents(self) -> List[Dict]:
        return [{"id": k, "title": v["title"]} for k, v in self.documents.items()]

vector_store = None

def initialize_vector_store():
    global vector_store
    vector_store = VectorStore()

def get_vector_store() -> VectorStore:
    return vector_store

import re
import pickle
import numpy as np
from pathlib import Path
from typing import List
from collections import Counter

STORE_PATH = Path("data/vector_store.pkl")


class VectorStore:
    def __init__(self):
        self.documents: List[dict] = []
        self._load()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _vector(self, tokens: List[str], vocab: List[str]) -> np.ndarray:
        count = Counter(tokens)
        vec = np.array([count.get(w, 0) for w in vocab], dtype=float)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def add(self, text: str, source: str, chunk_id: int):
        self.documents.append({"text": text, "source": source, "chunk_id": chunk_id, "tokens": self._tokenize(text)})
        self._save()

    def search(self, query: str, top_k: int = 4) -> List[dict]:
        if not self.documents:
            return []
        q_tokens = self._tokenize(query)
        vocab = sorted({t for doc in self.documents for t in doc["tokens"]} | set(q_tokens))
        q_vec = self._vector(q_tokens, vocab)
        scored = sorted(
            [(float(np.dot(q_vec, self._vector(doc["tokens"], vocab))), doc) for doc in self.documents],
            reverse=True
        )
        return [doc for score, doc in scored[:top_k] if score > 0]

    def sources(self) -> List[str]:
        return list({doc["source"] for doc in self.documents})

    def total_chunks(self) -> int:
        return len(self.documents)

    def clear(self):
        self.documents = []
        self._save()

    def _save(self):
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STORE_PATH, "wb") as f:
            pickle.dump(self.documents, f)

    def _load(self):
        if STORE_PATH.exists():
            with open(STORE_PATH, "rb") as f:
                self.documents = pickle.load(f)


_store = VectorStore()

def get_store() -> VectorStore:
    return _store

import json
import re
from pathlib import Path
from typing import List
from agents import function_tool
from core.vector_store import get_store

_store = get_store()


def _chunk(text: str) -> List[str]:
    words = text.split()
    chunks, step = [], 450
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + 500])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


@function_tool
def upload_document(file_path: str) -> str:
    """Index a .txt or .pdf file into the vector store."""
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"error": f"File not found: {file_path}"})
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix.lower() == ".pdf":
            import PyPDF2
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join(p.extract_text() or "" for p in reader.pages)
        else:
            return json.dumps({"error": "Only .txt and .pdf supported"})
        if not text.strip():
            return json.dumps({"error": "Document is empty"})
        chunks = _chunk(text)
        for i, chunk in enumerate(chunks):
            _store.add(chunk, path.name, i)
        return json.dumps({"status": "indexed", "file": path.name, "chunks": len(chunks)})
    except Exception as e:
        return json.dumps({"error": str(e)})


@function_tool
def add_text_directly(text: str, source_name: str) -> str:
    """Index raw text directly without a file."""
    try:
        chunks = _chunk(text)
        for i, chunk in enumerate(chunks):
            _store.add(chunk, source_name, i)
        return json.dumps({"status": "indexed", "source": source_name, "chunks": len(chunks)})
    except Exception as e:
        return json.dumps({"error": str(e)})


@function_tool
def search_documents(query: str, top_k: int = 4) -> str:
    """Search indexed documents for relevant context."""
    try:
        results = _store.search(query, top_k=top_k)
        if not results:
            return json.dumps({"message": "No results. Index documents first.", "results": []})
        return json.dumps({"results": [{"source": r["source"], "chunk_id": r["chunk_id"], "text": r["text"][:500]} for r in results]})
    except Exception as e:
        return json.dumps({"error": str(e)})


@function_tool
def list_indexed_documents() -> str:
    """List all indexed documents."""
    return json.dumps({"documents": _store.sources(), "total_chunks": _store.total_chunks()})


@function_tool
def clear_vector_store() -> str:
    """Clear all documents from the vector store."""
    _store.clear()
    return json.dumps({"status": "cleared"})

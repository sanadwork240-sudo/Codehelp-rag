"""Parse documents -> chunk -> embed -> store in ChromaDB.

Run from the project folder:
    python ingest.py            # ingest everything in data/raw
    python ingest.py --reset    # wipe the vector DB first, then ingest
"""
import argparse
from functools import lru_cache
from pathlib import Path

import argparse
import re
from functools import lru_cache
from pathlib import Path


import chromadb
import docx
import pymupdf as fitz  # PyMuPDF
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

SUPPORTED = {".pdf", ".docx", ".txt"}



@lru_cache(maxsize=1)
def get_collection():
    """One shared Chroma collection (the embedding model loads only once)."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBED_MODEL)
    return client.get_or_create_collection(
        name=config.COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},  # so score = 1 - distance is a cosine similarity
    )


def load_pages(path: Path):
    """Return a list of (page_number, text). Page numbers are 1-based."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        with fitz.open(path) as pdf:
            return [(i + 1, page.get_text("text")) for i, page in enumerate(pdf)]
    if ext == ".docx":
        d = docx.Document(path)
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return [(1, "\n".join(parts))]
    if ext == ".txt":
        return [(1, path.read_text(encoding="utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {ext}")

def doc_title(path: Path) -> str:
    """'02_Fee_Structure.pdf' -> 'Fee Structure'"""
    return re.sub(r"^\d+_", "", path.stem).replace("_", " ")

def ingest_file(path) -> int:
    """Ingest one file. Re-ingesting the same file replaces its old chunks. Returns #chunks."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {path.suffix} (use PDF, DOCX or TXT)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    ids, docs, metas = [], [], []
    for page_no, text in load_pages(path):
        if not text.strip():
            continue  # blank or scanned page (would need OCR)
        for i, chunk in enumerate(splitter.split_text(text)):
            ids.append(f"{path.name}-p{page_no}-c{i}")
            docs.append(f"{doc_title(path)}\n{chunk}")
            metas.append({"source": path.name, "page": page_no})

    if not docs:
        raise ValueError(f"No readable text found in {path.name} (scanned PDF?).")

    col = get_collection()
    col.delete(where={"source": path.name})          # avoid duplicates on re-ingest
    col.add(ids=ids, documents=docs, metadatas=metas)
    return len(docs)


def list_sources():
    data = get_collection().get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete all existing chunks first")
    args = ap.parse_args()

    if args.reset:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        try:
            client.delete_collection(config.COLLECTION)
            print("Vector DB cleared.")
        except Exception:
            pass
        get_collection.cache_clear()

    files = [f for f in sorted(config.RAW_DIR.glob("*")) if f.suffix.lower() in SUPPORTED]
    if not files:
        print(f"No documents found in {config.RAW_DIR}")
        return
    total = 0
    for f in files:
        try:
            n = ingest_file(f)
            total += n
            print(f"  OK  {f.name}: {n} chunks")
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
    print(f"Done. {total} chunks stored in {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()

"""Creation and persistence of the BM25 corpus used by hybrid retrieval."""
from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from config.settings import BM25_INDEX_PATH
from chunking.character_splitter import Chunk


@dataclass
class PersistedBM25Index:
    bm25: BM25Okapi
    ids: List[str]
    documents: List[str]
    metadatas: List[dict]


def save_bm25_index(chunks: List[Chunk], path: str = BM25_INDEX_PATH) -> None:
    ids = [f"{chunk.metadata['doc_title']}_{chunk.metadata['chunk_index']}" for chunk in chunks]
    documents = [chunk.content for chunk in chunks]
    index = PersistedBM25Index(
        bm25=BM25Okapi([document.lower().split() for document in documents]),
        ids=ids,
        documents=documents,
        metadatas=[dict(chunk.metadata) for chunk in chunks],
    )
    with Path(path).open("wb") as handle:
        pickle.dump(index, handle)
    print(f"Persisted BM25 index: ./{path}")


def load_bm25_index(path: str = BM25_INDEX_PATH) -> PersistedBM25Index:
    index_path = Path(path)
    if not index_path.exists():
        raise RuntimeError("BM25 index not found. Run an ingestion script first.")
    with index_path.open("rb") as handle:
        return pickle.load(handle)

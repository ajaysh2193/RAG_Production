"""Cross-encoder reranking of the candidate chunks returned by hybrid search."""
from typing import List

from sentence_transformers import CrossEncoder

from retrieval.semantic_search import RetrievedChunk


def rerank(query: str, chunks: List[RetrievedChunk], model_name: str, top_k: int) -> List[RetrievedChunk]:
    if not chunks:
        return []
    model = CrossEncoder(model_name)
    scores = model.predict([(query, chunk.content) for chunk in chunks])
    ordered = sorted(zip(chunks, scores), key=lambda item: float(item[1]), reverse=True)[:top_k]
    return [RetrievedChunk(content=chunk.content, metadata=chunk.metadata, score=float(score), rank=rank)
            for rank, (chunk, score) in enumerate(ordered, start=1)]

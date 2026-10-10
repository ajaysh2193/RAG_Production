import json
import os
from typing import List, Sequence

import boto3

from chunking.character_splitter import Chunk
from config.settings import AWS_REGION


class BedrockEmbeddingModel:
    """Small adapter that gives Amazon Titan the ``encode`` API used by RAG."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", AWS_REGION))

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True}),
                contentType="application/json",
                accept="application/json",
            )
            vectors.append(json.loads(response["body"].read())["embedding"])
        return vectors


def load_embedding_model(model_name: str) -> BedrockEmbeddingModel:
    """Return an Amazon Bedrock Titan embedding client."""
    return BedrockEmbeddingModel(model_name)


def embed_chunks(chunks: List[Chunk], model: BedrockEmbeddingModel) -> List[List[float]]:
    """
    Embed a list of Chunk objects and return a list of embedding vectors.
    Each vector corresponds to the Chunk at the same index.
    Prints embedding progress: 'Embedding X chunks with model Y...'
    """
    print(f"Embedding {len(chunks)} chunks with Bedrock model {model.model_id}...")
    texts = [chunk.content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings

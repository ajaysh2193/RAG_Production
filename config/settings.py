import os

from dotenv import load_dotenv

load_dotenv()

DOCS_PATH = "docs"
CHROMA_PATH = "chroma_data"
COLLECTION_NAME = "rag_docs"
BM25_INDEX_PATH = "bm25_index.pkl"

# AWS Bedrock.  These can be overridden in .env for the AWS Region and model
# IDs enabled in your account.
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "amazon.titan-embed-text-v2:0")
BEDROCK_GENERATION_MODEL = os.getenv("BEDROCK_GENERATION_MODEL", "amazon.nova-lite-v1:0")

# Chunking
CHAR_CHUNK_SIZE = 500
CHAR_CHUNK_OVERLAP = 50
SEMANTIC_MAX_WORDS = 400      # sections above this trigger H3 splitting

# Retrieval
TOP_K = 5
RERANK_CANDIDATES = 10
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

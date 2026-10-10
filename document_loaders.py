"""Load supported source documents as Markdown-like text before chunking."""
from pathlib import Path


def load_document(path: str) -> str:
    source = Path(path)
    if source.suffix.lower() == ".md":
        return source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".pdf":
        try:
            import pymupdf4llm
        except ImportError as exc:
            raise RuntimeError("PDF support requires pymupdf4llm. Run pip install -r requirements.txt.") from exc
        return pymupdf4llm.to_markdown(str(source))
    raise ValueError(f"Unsupported document type: {source.suffix}")

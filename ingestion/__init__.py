from ingestion.chunker import Chunk, TokenChunker
from ingestion.embedder import build_embedder


def __getattr__(name: str):
    if name in ("IngestionPipeline", "build_ingestion_pipeline"):
        from ingestion.pipeline import IngestionPipeline, build_ingestion_pipeline

        return {
            "IngestionPipeline": IngestionPipeline,
            "build_ingestion_pipeline": build_ingestion_pipeline,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TokenChunker",
    "Chunk",
    "build_embedder",
    "IngestionPipeline",
    "build_ingestion_pipeline",
]

"""
main.py
End-to-end Phase 1 demo.

Usage:
    # With OpenAI (full RAG):
    OPENAI_API_KEY=sk-... python main.py

    # Without API key (retrieval-only mode):
    python main.py --retrieval-only
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()  


SAMPLE_DOCS = [
    {
        "source": "ai_overview.txt",
        "text": (
            "Artificial intelligence (AI) is intelligence demonstrated by machines, "
            "as opposed to the natural intelligence displayed by animals including humans. "
            "AI research has been defined as the field of study of intelligent agents, "
            "which refers to any system that perceives its environment and takes actions "
            "that maximize its chance of achieving its goals. "
            * 30  
        ),
        "metadata": {"topic": "AI overview", "year": "2024"},
    },
    {
        "source": "rag_explained.txt",
        "text": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines information "
            "retrieval with language model generation. Instead of relying solely on the "
            "parametric knowledge stored in model weights, RAG first retrieves relevant "
            "documents from an external knowledge base and conditions the generation on "
            "those documents. This reduces hallucination and keeps answers grounded. "
            "The retrieval step typically uses dense vector search (semantic search) "
            "over embeddings stored in a vector database such as Chroma or Weaviate. "
            * 25
        ),
        "metadata": {"topic": "RAG", "year": "2024"},
    },
    {
        "source": "vector_databases.txt",
        "text": (
            "Vector databases are purpose-built systems for storing and querying "
            "high-dimensional embedding vectors. Popular options include Chroma, Weaviate, "
            "Pinecone, Qdrant, and Milvus. They support approximate nearest-neighbour "
            "(ANN) search algorithms such as HNSW (Hierarchical Navigable Small World) "
            "to make similarity search fast at scale. ChromaDB is particularly suitable "
            "for local development and prototyping because it requires no external "
            "infrastructure and persists data to disk automatically. "
            * 25
        ),
        "metadata": {"topic": "Vector DBs", "year": "2024"},
    },
]


def main():
    parser = argparse.ArgumentParser(description="Phase 1 RAG Pipeline Demo")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM answer generation (no API key needed)",
    )
    parser.add_argument(
        "--config",
        default="config/phase1.yaml",
        help="Path to config YAML (default: config/phase1.yaml)",
    )
    args = parser.parse_args()

    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    def print_section(title: str, text: str = "") -> None:
        if use_rich:
            console.rule(f"[bold cyan]{title}[/bold cyan]")
            if text:
                console.print(text)
        else:
            print(f"\n{'─'*60}\n{title}\n{'─'*60}")
            if text:
                print(text)

    print_section("Building pipeline", f"Config: {args.config}")

    from ingestion.pipeline import build_ingestion_pipeline
    ingestion, bundle = build_ingestion_pipeline(args.config)

    print_section("Ingesting documents")
    stats = ingestion.ingest_documents(SAMPLE_DOCS)
    print(f"  ✓ {stats}")

    test_query = "How does RAG reduce hallucination in language models?"
    print_section("Retrieval", f"Query: {test_query!r}")

    result = bundle.retriever.retrieve(test_query)
    for i, chunk in enumerate(result.chunks):
        if use_rich:
            console.print(
                f"  [green]{i+1}.[/green] [{chunk.citation_id}] "
                f"score={chunk.score:.4f}  source={chunk.source}"
            )
        else:
            print(f"  {i+1}. [{chunk.citation_id}] score={chunk.score:.4f}  source={chunk.source}")

    if not args.retrieval_only:
        if not os.getenv("OPENAI_API_KEY"):
            print(
                "\nNo OPENAI_API_KEY found — skipping generation. "
                "Run with --retrieval-only to suppress this message."
            )
        else:
            print_section("Generating answer")
            answer = bundle.rag.answer(test_query)
            answer.pretty_print()


if __name__ == "__main__":
    main()

"""
Hybrid retrieval over the indexed job chunks.

Retrieval is two-stage:
  1. Dense semantic search in ChromaDB (cosine similarity over MiniLM embeddings).
  2. A lightweight lexical-overlap re-rank on top of the candidate pool, so an
     exact keyword match (e.g. a specific tool or certification) can nudge a
     job up even if its overall semantic similarity is slightly lower.

This hybrid approach is a cheaper, dependency-light stand-in for a
cross-encoder re-ranker. If you want to swap in a real cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`), replace `calculate_lexical_overlap`
with a call to the cross-encoder's `.predict()` on (resume, chunk) pairs.
"""
import logging
import re
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import (
    COLLECTION_NAME,
    DEFAULT_INITIAL_POOL_K,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    LEXICAL_WEIGHT,
    VECTOR_STORE_DIR,
    VECTOR_WEIGHT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STOP_WORDS = {
    "and", "the", "with", "for", "that", "this", "from", "you", "will",
    "our", "are", "have", "role", "work", "experience", "years", "team",
}


def calculate_lexical_overlap(query_text: str, doc_text: str) -> float:
    """Jaccard-like overlap of technical tokens between query and document."""
    query_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", query_text.lower()))
    doc_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", doc_text.lower()))

    query_tokens -= STOP_WORDS
    doc_tokens -= STOP_WORDS

    if not query_tokens:
        return 0.0

    intersection = query_tokens.intersection(doc_tokens)
    return len(intersection) / len(query_tokens)


class JobRetriever:
    def __init__(
        self,
        persist_dir=VECTOR_STORE_DIR,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        if not persist_dir.exists():
            raise FileNotFoundError(
                f"Vector store not found at {persist_dir}. "
                "Run `python -m src.scrape_jobs && python -m src.preprocess && python -m src.embed` first."
            )

        logger.info("Initializing vector store client for retrieval...")
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Collection '{collection_name}' not found in vector store. "
                "Run `python -m src.embed` to build it."
            ) from e
        self.embedder = SentenceTransformer(model_name)

    def search_raw(self, query_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        query_vector = self.embedder.encode([query_text], convert_to_numpy=True).tolist()

        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        hits = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance range is [0, 2]; similarity is 1 - distance
            vector_sim = max(0.0, 1.0 - dist)
            hits.append({"text": doc, "metadata": meta, "vector_sim": vector_sim})
        return hits

    def retrieve_and_rerank(
        self,
        resume_text: str,
        top_k: int = DEFAULT_TOP_K,
        initial_pool_k: int = DEFAULT_INITIAL_POOL_K,
    ) -> list[dict[str, Any]]:
        """
        Retrieves candidate chunks, applies lexical re-ranking, and aggregates
        by unique job_id to return the top distinct jobs (not just top chunks,
        which could all belong to the same posting).
        """
        raw_hits = self.search_raw(resume_text, top_k=initial_pool_k)

        scored_hits = []
        for hit in raw_hits:
            lexical_sim = calculate_lexical_overlap(resume_text, hit["text"])
            vector_sim = hit["vector_sim"]

            composite_score = (VECTOR_WEIGHT * vector_sim) + (LEXICAL_WEIGHT * lexical_sim)

            hit["lexical_sim"] = lexical_sim
            hit["composite_score"] = round(composite_score, 4)
            scored_hits.append(hit)

        scored_hits.sort(key=lambda x: x["composite_score"], reverse=True)

        aggregated_jobs: dict[str, dict[str, Any]] = {}
        for hit in scored_hits:
            job_id = hit["metadata"].get("job_id")
            if job_id not in aggregated_jobs:
                aggregated_jobs[job_id] = {
                    "job_id": job_id,
                    "title": hit["metadata"].get("title"),
                    "company": hit["metadata"].get("company"),
                    "location": hit["metadata"].get("location"),
                    "job_url": hit["metadata"].get("job_url"),
                    "best_matching_chunk": hit["text"],
                    "chunk_type": hit["metadata"].get("chunk_type"),
                    "score": hit["composite_score"],
                    "vector_sim": round(hit["vector_sim"], 4),
                    "lexical_sim": round(hit["lexical_sim"], 4),
                }

            if len(aggregated_jobs) == top_k:
                break

        return list(aggregated_jobs.values())


if __name__ == "__main__":
    sample_resume = (
        "AI / ML Engineer with expertise in building scalable Retrieval-Augmented Generation (RAG) "
        "pipelines using LangChain, LangGraph, and ChromaDB. Proficient in Python, PyTorch, Hugging Face, "
        "and deploying containerized microservices using Docker, Kubernetes, and AWS."
    )

    logger.info("Running standalone retrieval test with sample ML resume...")
    retriever = JobRetriever()
    matched_jobs = retriever.retrieve_and_rerank(sample_resume, top_k=3)

    print("\n" + "=" * 60)
    print("STANDALONE RETRIEVAL RESULTS (Top Matches)")
    print("=" * 60)
    for idx, job in enumerate(matched_jobs, 1):
        print(f"\n[{idx}] {job['title']} at {job['company']}")
        print(f"    Location: {job['location']}")
        print(f"    Composite Match Score: {job['score']} (Vector: {job['vector_sim']}, Lexical: {job['lexical_sim']})")
        print(f"    Top Matched Section ({job['chunk_type']}):")
        print("    " + "\n    ".join(job["best_matching_chunk"].split("\n")[:4]) + "...")

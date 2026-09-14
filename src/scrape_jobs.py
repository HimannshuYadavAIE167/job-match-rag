"""
Collects raw job postings for the corpus.

Tries a live LinkedIn scrape via `jobspy` first. If that fails for any
reason (rate limiting, network block, missing dependency, schema change),
it falls back to a small built-in seed dataset so the rest of the pipeline
(preprocess -> embed -> retrieve -> generate) is never blocked by a flaky
scraper. Swap in your own dataset (e.g. a Kaggle/HuggingFace job-postings
dump) by writing directly to `data/raw/jobs.json` in the same schema.
"""
import json
import logging

from src.config import RAW_DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_mock_jobs() -> list[dict]:
    """Fallback dataset containing realistic, high-signal ML/AI job descriptions."""
    return [
        {
            "id": "mock_mle_001",
            "title": "Machine Learning Engineer",
            "company": "Cognitive Scale AI",
            "location": "Remote - US",
            "job_url": "https://example.com/jobs/001",
            "description": (
                "About the Role:\n"
                "We are seeking an experienced Machine Learning Engineer to build and optimize "
                "our core RAG and retrieval infrastructure. You will design vector search indices, "
                "fine-tune embedding models, and deploy real-time inference microservices.\n\n"
                "Key Responsibilities:\n"
                "- Architect scalable retrieval-augmented generation (RAG) pipelines using ChromaDB/Pinecone.\n"
                "- Fine-tune transformer models using PyTorch and Hugging Face Transformers.\n"
                "- Build high-throughput RESTful inference APIs with FastAPI and Docker.\n"
                "- Collaborate with data engineering to curate high-quality pre-training corpora.\n\n"
                "Requirements:\n"
                "- Strong proficiency in Python, PyTorch, and Scikit-learn.\n"
                "- 3+ years experience designing vector search or semantic retrieval systems.\n"
                "- Hands-on production experience with Docker, Kubernetes, and AWS (SageMaker, S3, ECS).\n"
                "- Solid grasp of cross-encoders, BM25 hybrid search, and LLM evaluation benchmarks."
            ),
        },
        {
            "id": "mock_genai_002",
            "title": "Generative AI Systems Engineer",
            "company": "Nexus Agentics",
            "location": "Remote",
            "job_url": "https://example.com/jobs/002",
            "description": (
                "Role Overview:\n"
                "Nexus Agentics is looking for an AI Engineer specialized in autonomous agents and multi-turn reasoning. "
                "You will lead the development of agentic orchestration pipelines for automated developer tools.\n\n"
                "Responsibilities:\n"
                "- Build multi-agent workflows using LangGraph, LangChain, and AutoGen.\n"
                "- Optimize prompt strategies, structured outputs, and tool-calling with Claude and OpenAI APIs.\n"
                "- Implement context pruning, summarization, and persistent state management.\n"
                "- Monitor hallucination rates and latency metrics using LangSmith or custom evals.\n\n"
                "Minimum Qualifications:\n"
                "- B.Tech/M.S. in Computer Science or equivalent hands-on experience.\n"
                "- Mastery of Python, async programming, and API integration.\n"
                "- Proven portfolio demonstrating multi-agent workflows and tool-calling execution."
            ),
        },
        {
            "id": "mock_de_003",
            "title": "Data Engineer - Vector Pipelines",
            "company": "StreamData Inc.",
            "location": "Remote",
            "job_url": "https://example.com/jobs/003",
            "description": (
                "What You Will Do:\n"
                "As a Data Engineer, you will own our extract-transform-load (ETL) pipelines feeding our semantic search databases. "
                "You will process millions of unstructured documents into tokenized embeddings.\n\n"
                "Core Responsibilities:\n"
                "- Develop distributed data pipelines with Apache Spark and Python.\n"
                "- Manage database schemas and vector collections across PostgreSQL (pgvector) and MongoDB.\n"
                "- Maintain CI/CD workflows using GitHub Actions and Terraform.\n\n"
                "Requirements:\n"
                "- Advanced SQL and Python skills.\n"
                "- Experience handling high-volume unstructured text pipelines.\n"
                "- Familiarity with AWS or GCP cloud storage architectures."
            ),
        },
        {
            "id": "mock_be_004",
            "title": "Senior Backend Engineer (Python / Cloud)",
            "company": "CloudForge Labs",
            "location": "Remote",
            "job_url": "https://example.com/jobs/004",
            "description": (
                "Overview:\n"
                "Looking for a Senior Backend Engineer to maintain core microservices powering AI workloads.\n\n"
                "Responsibilities:\n"
                "- Build scalable, observable APIs using Python, FastAPI, and Redis queues.\n"
                "- Manage infrastructure-as-code using Terraform and Docker containers.\n"
                "- Implement authentication, rate limiting, and database connection pooling.\n\n"
                "Qualifications:\n"
                "- 4+ years of backend engineering in Python or Go.\n"
                "- Deep understanding of distributed caching, PostgreSQL, and Linux internals."
            ),
        },
        {
            "id": "mock_fe_005",
            "title": "Frontend React / UI Developer",
            "company": "PixelCraft Studio",
            "location": "Remote",
            "job_url": "https://example.com/jobs/005",
            "description": (
                "Responsibilities:\n"
                "- Design and implement rich UI components using React, TypeScript, and Tailwind CSS.\n"
                "- Build real-time client state management and websocket integrations.\n\n"
                "Qualifications:\n"
                "- Proficient in modern JavaScript/TypeScript, React 18, Next.js.\n"
                "- Strong eye for UX design and accessibility (WCAG)."
            ),
        },
    ]


def fetch_and_save_jobs(
    search_term: str = "Machine Learning Engineer",
    location: str = "Remote",
    results_wanted: int = 15,
):
    """Scrape (or fall back to seed data) and persist raw postings to disk."""
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []

    try:
        from jobspy import scrape_jobs

        logger.info(f"Attempting live scrape on LinkedIn for '{search_term}'...")
        jobs_df = scrape_jobs(
            site_name=["linkedin"],  # Omit Indeed to prevent 403 blocks
            search_term=search_term,
            location=location,
            results_wanted=results_wanted,
            is_remote=True,
        )
        if not jobs_df.empty:
            keep_cols = ["id", "title", "company", "location", "description", "job_url"]
            existing_cols = [c for c in keep_cols if c in jobs_df.columns]
            jobs_df = jobs_df[existing_cols].dropna(subset=["description"])
            records = jobs_df.to_dict(orient="records")
    except Exception as e:
        logger.warning(f"Live scraping failed or blocked ({e}). Switching to mock seed dataset.")

    if not records:
        logger.info("Using built-in seed dataset to unblock the pipeline.")
        records = get_mock_jobs()

    with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(records)} job postings to {RAW_DATA_PATH}")
    return RAW_DATA_PATH


if __name__ == "__main__":
    fetch_and_save_jobs()

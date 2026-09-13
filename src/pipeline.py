import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path regardless of how the script is invoked
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.generate import MatchExplainer
    from src.retrieve import JobRetriever
except ImportError:
    from generate import MatchExplainer
    from retrieve import JobRetriever
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JobMatchPipeline:
    def __init__(
        self,
        retriever: JobRetriever | None = None,
        explainer: MatchExplainer | None = None
    ):
        logger.info("Initializing JobMatchPipeline components...")
        self.retriever = retriever or JobRetriever()
        self.explainer = explainer or MatchExplainer()

    def run(
        self,
        resume_text: str,
        top_k: int = 3,
        generate_explanations: bool = True
    ) -> list[dict[str, Any]]:
        """
        Executes the full RAG pipeline:
        Resume Text -> Hybrid Retrieval & Re-ranking -> Gemini Gap Analysis
        """
        clean_resume = resume_text.strip()
        if not clean_resume:
            logger.warning("Empty resume text supplied to pipeline.")
            return []

        logger.info(f"Retrieving top {top_k} matching jobs for input profile...")
        matched_jobs = self.retriever.retrieve_and_rerank(
            resume_text=clean_resume,
            top_k=top_k
        )

        if not matched_jobs:
            logger.info("No matching jobs retrieved.")
            return []

        results = []
        for idx, job in enumerate(matched_jobs, 1):
            logger.info(f"Processing candidate match {idx}/{len(matched_jobs)}: {job['title']} at {job['company']}")

            explanation = {}
            if generate_explanations:
                explanation = self.explainer.explain_match(
                    resume_text=clean_resume,
                    job_title=job["title"],
                    company=job["company"],
                    job_chunk=job["best_matching_chunk"],
                    match_score=job["score"]
                )

            results.append({
                "rank": idx,
                "job_id": job["job_id"],
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "job_url": job["job_url"],
                "match_score": job["score"],
                "vector_similarity": job["vector_sim"],
                "lexical_overlap": job["lexical_sim"],
                "matched_context": job["best_matching_chunk"],
                "explanation": explanation
            })

        return results


def match_resume(resume_text: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Convenience functional wrapper."""
    pipeline = JobMatchPipeline()
    return pipeline.run(resume_text, top_k=top_k)


if __name__ == "__main__":
    test_resume = """
    Software Engineer with hands-on focus in Artificial Intelligence and Machine Learning.
    - Designed and implemented Retrieval-Augmented Generation (RAG) frameworks with vector databases (ChromaDB, Pinecone).
    - Proficient with Python, PyTorch, Scikit-learn, LangChain, and LangGraph multi-agent orchestration.
    - Built containerized backend microservices using FastAPI, Docker, and deployed on cloud infrastructure (AWS).
    """

    print("\n--- Running End-to-End Pipeline Smoke Test ---")
    pipeline = JobMatchPipeline()
    matches = pipeline.run(test_resume, top_k=2)

    for item in matches:
        print(f"\n[{item['rank']}] {item['title']} - {item['company']}")
        print(f"Overall Fit: {round(item['match_score'] * 100, 1)}% | Location: {item['location']}")
        exp = item["explanation"]
        print(f"Rationale: {exp.get('match_rationale', 'N/A')}")
        print("Strengths:", exp.get("matching_strengths", []))
        print("Skill Gaps:", exp.get("skill_gaps", []))
        print("Recommendations:", exp.get("actionable_recommendations", []))
"""
Benchmarks retrieval quality (Precision@K, Recall@K, MRR) against the
labeled resume-job pairs in eval/labeled_pairs.csv.
"""
import csv
import logging

from src.config import EVAL_RESULTS_PATH, LABELED_PAIRS_PATH
from src.retrieve import JobRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_metrics_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 3) -> dict[str, float]:
    """Calculates Precision@K, Recall@K, and Reciprocal Rank."""
    top_k_retrieved = retrieved_ids[:k]
    hits = [doc_id for doc_id in top_k_retrieved if doc_id in relevant_ids]

    precision = len(hits) / k if k > 0 else 0.0
    recall = len(hits) / len(relevant_ids) if relevant_ids else 0.0

    reciprocal_rank = 0.0
    for rank_idx, doc_id in enumerate(top_k_retrieved, start=1):
        if doc_id in relevant_ids:
            reciprocal_rank = 1.0 / rank_idx
            break

    return {
        f"precision@{k}": round(precision, 4),
        f"recall@{k}": round(recall, 4),
        "reciprocal_rank": round(reciprocal_rank, 4),
    }


def run_evaluation(k: int = 3) -> None:
    if not LABELED_PAIRS_PATH.exists():
        raise FileNotFoundError(
            f"Missing evaluation dataset: {LABELED_PAIRS_PATH}. Run eval/generate_ground_truth.py first."
        )

    logger.info("Initializing Retriever for evaluation benchmark...")
    retriever = JobRetriever()

    with open(LABELED_PAIRS_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(rows)} labeled test pairs from {LABELED_PAIRS_PATH}")

    precisions, recalls, mrr_list, detailed_logs = [], [], [], []

    for row in rows:
        r_id = row["resume_id"]
        resume_text = row["resume_text"]
        relevant_ids = {x.strip() for x in row["relevant_job_ids"].split(",") if x.strip()}

        matched_jobs = retriever.retrieve_and_rerank(resume_text, top_k=k)
        retrieved_ids = [j["job_id"] for j in matched_jobs]

        metrics = compute_metrics_at_k(retrieved_ids, relevant_ids, k=k)
        precisions.append(metrics[f"precision@{k}"])
        recalls.append(metrics[f"recall@{k}"])
        mrr_list.append(metrics["reciprocal_rank"])

        detailed_logs.append(
            f"Resume: {r_id}\n"
            f"  Expected: {relevant_ids}\n"
            f"  Retrieved: {retrieved_ids}\n"
            f"  P@{k}: {metrics[f'precision@{k}']:.2f} | R@{k}: {metrics[f'recall@{k}']:.2f} | RR: {metrics['reciprocal_rank']:.2f}\n"
        )

    mean_precision = sum(precisions) / len(precisions) if precisions else 0.0
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mean_mrr = sum(mrr_list) / len(mrr_list) if mrr_list else 0.0

    summary = (
        "=" * 60 + "\n"
        f"RETRIEVAL EVALUATION SUMMARY (K={k})\n"
        "=" * 60 + "\n"
        f"Total Test Personas: {len(rows)}\n"
        f"Mean Precision@{k}: {mean_precision:.4f}\n"
        f"Mean Recall@{k}:    {mean_recall:.4f}\n"
        f"Mean MRR:           {mean_mrr:.4f}\n"
        "=" * 60 + "\n\n"
        "Persona Breakdown:\n" + "\n".join(detailed_logs)
    )

    print("\n" + summary)

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(summary)

    logger.info(f"Evaluation logs written to {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation(k=3)

"""
Builds a small labeled evaluation set (resume persona -> relevant job IDs)
from whatever is currently indexed in data/processed/chunks.json.

BUGFIX vs. the original version: matching used to pick the first N jobs
whose title contained ANY keyword from a persona's keyword list, in
dict-insertion order. Because keyword lists included generic terms like
"engineer" or "ai" that appear in almost every mock job title, every
persona ended up matched to the same first two jobs regardless of role
(e.g. the Data Engineer persona was silently matched to the ML Engineer
and GenAI Engineer postings). This version scores each job by how many
keywords it matches and ranks by that score, so a persona is matched to
the jobs that are actually most relevant to it.
"""
import csv
import json
import sys
from pathlib import Path

# Insert project root into sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import from src cleanly
from src.config import LABELED_PAIRS_PATH, PROCESSED_DATA_PATH    
with open(PROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Extract unique jobs, preserving first-seen order.
jobs: dict[str, dict] = {}
for c in chunks:
    meta = c["metadata"]
    j_id = meta["job_id"]
    if j_id not in jobs:
        jobs[j_id] = {
            "title": str(meta.get("title", "")).lower(),
            "company": str(meta.get("company", "")),
        }

print("\nIndexed Jobs in Corpus:")
for j_id, data in jobs.items():
    print(f"  [{j_id}] {data['title']} @ {data['company']}")


def rank_matching_job_ids(keywords: list[str], top_n: int = 2) -> list[str]:
    """
    Score every job by the number of distinct persona keywords found in its
    title, then return the top `top_n` job_ids by score (ties broken by
    corpus order). Jobs scoring 0 are excluded.
    """
    scored = []
    for j_id, data in jobs.items():
        score = sum(1 for kw in keywords if kw in data["title"])
        if score > 0:
            scored.append((score, j_id))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [j_id for _, j_id in scored[:top_n]]


# Keyword lists are ordered from most to least specific; specificity matters
# more than list length because the scorer just counts hits, so avoid overly
# generic single words (e.g. "engineer", "ai") that appear in most titles.
personas = [
    {
        "resume_id": "res_mle_01",
        "resume_text": "Machine Learning Engineer experienced in Python, PyTorch, RAG architectures, vector databases like ChromaDB, and Docker containerization.",
        "keywords": ["machine learning engineer", "machine learning", "ml"],
    },
    {
        "resume_id": "res_genai_02",
        "resume_text": "AI Engineer focused on autonomous multi-agent workflows using LangChain, LangGraph, prompt optimization, and Claude/OpenAI APIs.",
        "keywords": ["generative ai", "generative", "agentics", "systems engineer"],
    },
    {
        "resume_id": "res_de_03",
        "resume_text": "Data Engineer skilled in distributed ETL pipelines using Apache Spark, SQL, MongoDB, and AWS cloud storage architectures.",
        "keywords": ["data engineer", "vector pipelines", "etl"],
    },
    {
        "resume_id": "res_be_04",
        "resume_text": "Senior Backend Developer with hands-on experience in Python, FastAPI microservices, Redis queues, and Docker infrastructure.",
        "keywords": ["backend engineer", "backend", "python / cloud"],
    },
    {
        "resume_id": "res_fe_05",
        "resume_text": "Frontend Developer proficient in React 18, TypeScript, Tailwind CSS, and building responsive client-side user interfaces.",
        "keywords": ["frontend", "react", "ui developer"],
    },
]

rows = []
all_keys = list(jobs.keys())

for p in personas:
    matched_ids = rank_matching_job_ids(p["keywords"], top_n=2)
    if not matched_ids:
        matched_ids = [all_keys[0]]

    rows.append(
        {
            "resume_id": p["resume_id"],
            "resume_text": p["resume_text"],
            "relevant_job_ids": ",".join(matched_ids),
        }
    )

with open(LABELED_PAIRS_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["resume_id", "resume_text", "relevant_job_ids"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nGenerated aligned {LABELED_PAIRS_PATH} successfully.")

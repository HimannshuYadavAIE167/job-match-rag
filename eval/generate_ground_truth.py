import csv
import json
from pathlib import Path

CHUNKS_FILE = Path("data/processed/chunks.json")
OUTPUT_CSV = Path("eval/labeled_pairs.csv")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Extract unique jobs
jobs = {}
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


def find_matching_job_ids(keywords: list[str]) -> list[str]:
    matches = []
    for j_id, data in jobs.items():
        if any(kw in data["title"] for kw in keywords):
            matches.append(j_id)
    return matches


personas = [
    {
        "resume_id": "res_mle_01",
        "resume_text": "Machine Learning Engineer experienced in Python, PyTorch, RAG architectures, vector databases like ChromaDB, and Docker containerization.",
        "keywords": ["machine learning", "ml", "ai", "artificial intelligence", "data scientist"]
    },
    {
        "resume_id": "res_genai_02",
        "resume_text": "AI Engineer focused on autonomous multi-agent workflows using LangChain, LangGraph, prompt optimization, and Claude/OpenAI APIs.",
        "keywords": ["agent", "generative", "llm", "ai", "machine learning"]
    },
    {
        "resume_id": "res_de_03",
        "resume_text": "Data Engineer skilled in distributed ETL pipelines using Apache Spark, SQL, MongoDB, and AWS cloud storage architectures.",
        "keywords": ["data engineer", "etl", "pipeline", "data", "engineer"]
    },
    {
        "resume_id": "res_be_04",
        "resume_text": "Senior Backend Developer with hands-on experience in Python, FastAPI microservices, Redis queues, and Docker infrastructure.",
        "keywords": ["backend", "software engineer", "developer", "python", "systems"]
    },
    {
        "resume_id": "res_fe_05",
        "resume_text": "Frontend Developer proficient in React 18, TypeScript, Tailwind CSS, and building responsive client-side user interfaces.",
        "keywords": ["frontend", "react", "ui", "web", "full stack"]
    }
]

rows = []
all_keys = list(jobs.keys())

for p in personas:
    matched_ids = find_matching_job_ids(p["keywords"])
    if not matched_ids:
        matched_ids = [all_keys[0]]

    rows.append({
        "resume_id": p["resume_id"],
        "resume_text": p["resume_text"],
        "relevant_job_ids": ",".join(matched_ids[:2])
    })

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["resume_id", "resume_text", "relevant_job_ids"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nGenerated aligned {OUTPUT_CSV} successfully.")
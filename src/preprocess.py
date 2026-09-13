import json
import logging
import re
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RAW_FILE = Path("data/raw/jobs.json")
PROCESSED_FILE = Path("data/processed/chunks.json")


def clean_text(text: str) -> str:
    """Normalize whitespace and strip HTML/control artifacts."""
    if not text:
        return ""
    # Strip HTML tags if any leaked through scraping
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize special dashes, bullets, and spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\xa0]", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_sections(description: str) -> dict[str, str]:
    """
    Heuristically splits a job description into:
    - role_overview
    - responsibilities
    - requirements
    """
    sections = {
        "role_overview": "",
        "responsibilities": "",
        "requirements": ""
    }

    resp_headers = r"(responsibilities|what you will do|what you'll do|core tasks|key duties|the role)"
    req_headers = r"(requirements|qualifications|what you need|must have|skills|experience required)"

    pattern = re.compile(
        rf"(?P<resp>\b{resp_headers}\b[\s\S]*?)?(?P<req>\b{req_headers}\b[\s\S]*)?",
        re.IGNORECASE
    )

    lines = description.split("\n")
    current_section = "role_overview"
    section_buffers = {"role_overview": [], "responsibilities": [], "requirements": []}

    for line in lines:
        lower_line = line.strip().lower()
        if re.match(rf"^{resp_headers}[:\s]*$", lower_line):
            current_section = "responsibilities"
            continue
        elif re.match(rf"^{req_headers}[:\s]*$", lower_line):
            current_section = "requirements"
            continue

        section_buffers[current_section].append(line)

    for sec, buf in section_buffers.items():
        sections[sec] = clean_text("\n".join(buf))

    # Fallback: if headers weren't found, keep entire cleaned body in role_overview
    if not sections["responsibilities"] and not sections["requirements"]:
        sections["role_overview"] = clean_text(description)

    return sections


def create_chunks_from_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Produce semantic chunk documents with structured metadata."""
    job_id = str(job.get("id", ""))
    title = job.get("title", "Unknown Title")
    company = job.get("company", "Unknown Company")
    location = job.get("location", "Remote")
    job_url = job.get("job_url", "")
    description = job.get("description", "")

    cleaned_desc = clean_text(description)
    sections = extract_sections(cleaned_desc)

    chunks = []
    chunk_index = 0

    base_metadata = {
        "job_id": job_id,
        "title": title,
        "company": company,
        "location": location,
        "job_url": job_url,
    }

    # 1. Full context / Summary chunk
    summary_text = (
        f"Job Title: {title}\n"
        f"Company: {company}\n"
        f"Location: {location}\n"
        f"Summary: {sections['role_overview'][:500]}"
    )
    chunks.append({
        "chunk_id": f"{job_id}_chk_{chunk_index}",
        "text": summary_text,
        "metadata": {**base_metadata, "chunk_type": "overview"}
    })
    chunk_index += 1

    # 2. Responsibilities chunk (if present)
    if sections["responsibilities"]:
        resp_text = (
            f"Job Title: {title} at {company}\n"
            f"Responsibilities:\n{sections['responsibilities']}"
        )
        chunks.append({
            "chunk_id": f"{job_id}_chk_{chunk_index}",
            "text": resp_text,
            "metadata": {**base_metadata, "chunk_type": "responsibilities"}
        })
        chunk_index += 1

    # 3. Requirements & Qualifications chunk (critical for matching)
    if sections["requirements"]:
        req_text = (
            f"Job Title: {title} at {company}\n"
            f"Requirements & Qualifications:\n{sections['requirements']}"
        )
        chunks.append({
            "chunk_id": f"{job_id}_chk_{chunk_index}",
            "text": req_text,
            "metadata": {**base_metadata, "chunk_type": "requirements"}
        })
        chunk_index += 1

    return chunks


def process_all_jobs() -> Path:
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Input file {RAW_FILE} not found. Run scrape_jobs.py first.")

    with open(RAW_FILE, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    logger.info(f"Loaded {len(jobs)} raw jobs from {RAW_FILE}")

    all_chunks = []
    for job in jobs:
        chunks = create_chunks_from_job(job)
        all_chunks.extend(chunks)

    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Generated {len(all_chunks)} semantic chunks. Saved to {PROCESSED_FILE}")
    return PROCESSED_FILE


if __name__ == "__main__":
    process_all_jobs()
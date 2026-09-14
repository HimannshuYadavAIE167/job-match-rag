# JobMatch RAG — Smart Job-Matching Assistant

A retrieval-augmented assistant that takes a resume, retrieves the most
relevant job postings from an indexed corpus, and produces a grounded,
LLM-generated explanation of *why* each match works — matching skills,
missing skills, and a fit verdict — instead of a black-box similarity score.

**Live demo:** _add your Streamlit Community Cloud / HF Spaces link here after deploying_

## Why this is RAG and not just "ask an LLM"

The generation step is only ever shown the retrieved job chunk plus the
resume text, and is explicitly instructed not to invent skills that aren't
present in either. That constraint is what keeps the output grounded instead
of the model free-associating about the role.

## Architecture

```
Resume text
    │
    ▼
┌─────────────────────┐      ┌──────────────────────┐
│  1. Ingestion        │      │  Job postings         │
│  scrape_jobs.py      │─────▶│  (LinkedIn via jobspy, │
│  (falls back to seed │      │   or built-in mock     │
│   dataset if blocked)│      │   dataset)             │
└─────────────────────┘      └──────────────────────┘
    │
    ▼
┌─────────────────────┐
│  2. Preprocessing    │  Splits each posting into
│  preprocess.py       │  overview / responsibilities /
│                       │  requirements chunks
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  3. Embedding        │  all-MiniLM-L6-v2 sentence
│  embed.py             │  embeddings → ChromaDB
│                        │  (persistent, local)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  4. Hybrid retrieval  │  Cosine similarity search,
│  retrieve.py           │  re-ranked with a lexical
│                         │  keyword-overlap score
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  5. Generation        │  Gemini 2.5 Flash produces
│  generate.py            │  structured JSON: rationale,
│                          │  strengths, gaps, next steps
│                          │  (falls back to an offline
│                          │  heuristic explainer if no
│                          │  API key is set)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  6. App                │  Streamlit UI: paste a resume,
│  streamlit_app.py       │  see ranked matches + explanations
└─────────────────────┘
```

## Project layout

```
jobmatch-rag/
├── src/
│   ├── config.py         # paths, model names, API key resolution
│   ├── scrape_jobs.py     # step 1: ingestion
│   ├── preprocess.py      # step 2: chunking
│   ├── embed.py           # step 3: embeddings + ChromaDB indexing
│   ├── retrieve.py        # step 4: hybrid retrieval + re-ranking
│   ├── generate.py        # step 5: LLM gap-analysis generation
│   ├── pipeline.py        # wires retrieve.py + generate.py together
│   └── resume_parser.py   # extracts text from uploaded .pdf/.docx/.txt resumes
├── app/
│   └── streamlit_app.py   # step 6: the deployable UI
├── eval/
│   ├── generate_ground_truth.py  # builds labeled resume/job pairs
│   ├── labeled_pairs.csv          # generated output (checked in for reference)
│   └── evaluate.py                # Precision@K / Recall@K / MRR
├── tests/
│   └── test_preprocess.py  # offline unit tests (no model downloads needed)
├── data/                    # raw + processed corpus (gitignored, regenerated)
├── vector_store/            # ChromaDB persistence (gitignored, regenerated)
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

## Quickstart (local)

Requires Python 3.11+.

```bash
git clone <your-repo-url>
cd jobmatch-rag
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)
# the app runs fine without a key too — it just falls back to an offline
# keyword-overlap explainer instead of an LLM-generated one

# Build the corpus + vector index once:
python -m src.scrape_jobs      # scrapes LinkedIn via jobspy, or falls back to a seed dataset
python -m src.preprocess       # chunk the postings
python -m src.embed            # embed + index into ChromaDB

streamlit run app/streamlit_app.py
```

Or with `make`:

```bash
make install
make ingest       # scrape + preprocess + embed
make app
```

## Running with Docker

```bash
cp .env.example .env   # fill in GEMINI_API_KEY
docker compose up --build
```

The image builds the job corpus and vector index at *build time*, so the
container starts serving immediately with no cold-start delay.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at
   `app/streamlit_app.py`.
3. **Important:** Streamlit Cloud has no `.env` file. Under your app's
   **Settings → Secrets**, paste the contents of `.streamlit/secrets.toml.example`
   with your real key. `src/config.py` checks `st.secrets` first, so no code
   changes are needed.
4. Streamlit Cloud runs `pip install -r requirements.txt` automatically, but
   **the vector index still needs to be built** — either commit a pre-built
   `vector_store/` (simplest for a small demo corpus like this one) or add a
   one-time setup step. The app itself will show a clear error with the exact
   commands to run if the index is missing.

## Evaluation

Retrieval quality is benchmarked against a small labeled set of resume
personas mapped to the job postings they should match:

```bash
python -m eval.generate_ground_truth   # regenerate labels from current corpus
python -m eval.evaluate                 # Precision@K, Recall@K, MRR
```

**Note on the ground-truth generator:** the original version matched a
persona to a job if *any* keyword in its list appeared in the job title —
including generic terms like "engineer," which appears in nearly every
posting title in this corpus. That silently mapped every persona to the same
two jobs regardless of role. The fixed version (`eval/generate_ground_truth.py`)
scores each job by how many *specific* keywords match and ranks by that
score, so a "Data Engineer" persona is no longer mis-labeled as relevant to
an "ML Engineer" posting.

| Metric | Before re-ranker (vector-only) | After hybrid re-rank |
|---|---|---|
| Precision@3 | *fill in after running `evaluate.py` with `LEXICAL_WEIGHT=0` vs. default* | *fill in* |
| Recall@3 | | |
| MRR | | |

Run `evaluate.py` once with `LEXICAL_WEIGHT = 0.0` in `src/config.py` (vector-only
baseline) and once with the default `0.25` weight, and drop both result sets
into the table above — this before/after comparison is the single most useful
thing to show in a portfolio or interview, since it demonstrates the
re-ranking step is actually earning its complexity rather than just being
there for show.

## What's grounded vs. what's a known limitation

- Generation only ever sees the retrieved chunk + resume text, and is
  instructed not to invent skills — this is enforced by prompt instruction,
  not a hard constraint, so occasional drift is possible. The offline
  fallback explainer is fully deterministic and keyword-based.
- The lexical re-ranker is a lightweight stand-in for a cross-encoder. A real
  cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) would likely improve
  precision further at the cost of extra latency and a larger dependency —
  noted in `src/retrieve.py` as a drop-in swap point.
- `jobspy` scraping is best-effort; LinkedIn actively rate-limits and blocks
  scrapers, so the pipeline is designed to degrade gracefully to a seed
  dataset rather than fail outright.

## What I'd improve next

- Swap the lexical re-ranker for a real cross-encoder and quantify the
  precision delta.
- Expand the corpus beyond the 5-job seed dataset (or a larger scraped/Kaggle
  set) and re-run the evaluation at a more statistically meaningful scale.
- OCR fallback for scanned/image-based PDFs (current PDF parsing is text-layer
  only, and says so explicitly when it finds nothing extractable).
- Cache LLM explanations per (resume, job_id) pair to cut API costs.

## Tech stack

Python, sentence-transformers, ChromaDB, Google Gemini API, Streamlit, Docker, GitHub Actions.

"""
Centralized configuration for the JobMatch RAG pipeline.

Keeping paths, model names, and secret-resolution logic in one place means
every module (scrape/preprocess/embed/retrieve/generate + the Streamlit app)
agrees on where data lives and how to find API keys, whether the code is
running locally with a .env file or deployed on Streamlit Community Cloud
(which injects secrets via st.secrets, not environment variables).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "jobs.json"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "chunks.json"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

EVAL_DIR = PROJECT_ROOT / "eval"
LABELED_PAIRS_PATH = EVAL_DIR / "labeled_pairs.csv"
EVAL_RESULTS_PATH = EVAL_DIR / "eval_results.txt"

# ---------------------------------------------------------------------------
# Model / retrieval settings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME", "gemini-2.5-flash")
COLLECTION_NAME = "job_chunks"

DEFAULT_TOP_K = 3
DEFAULT_INITIAL_POOL_K = 15
VECTOR_WEIGHT = 0.75
LEXICAL_WEIGHT = 0.25


def get_api_key() -> str | None:
    """
    Resolve the Gemini API key from, in order of priority:
      1. Streamlit secrets (st.secrets["GEMINI_API_KEY"]) — used on Streamlit
         Community Cloud, where .env files don't exist.
      2. Environment variables (GEMINI_API_KEY, then legacy ANTHROPIC_API_KEY).

    Returns None if no usable key is found, so callers can fall back to the
    offline heuristic explainer instead of crashing.
    """
    try:
        import streamlit as st  # imported lazily so non-Streamlit contexts don't need it

        if hasattr(st, "secrets"):
            key = st.secrets.get("GEMINI_API_KEY")
            if key:
                return key
    except Exception:
        pass

    key = os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if key and not key.startswith("your_"):
        return key
    return None

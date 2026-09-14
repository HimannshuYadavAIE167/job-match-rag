@echo off
REM Run this AFTER: mkdir job-match-rag && cd job-match-rag
REM Usage: setup_empty_structure.bat
REM Creates all folders and empty files, ready for you to fill in.

REM Directories
mkdir data\raw
mkdir data\processed
mkdir data\resumes
mkdir src
mkdir eval
mkdir app
mkdir notebooks
mkdir vector_store

REM src/ files
type nul > src\__init__.py
type nul > src\scrape_jobs.py
type nul > src\preprocess.py
type nul > src\embed.py
type nul > src\retrieve.py
type nul > src\generate.py
type nul > src\pipeline.py

REM eval/ files
type nul > eval\labeled_pairs.csv
type nul > eval\evaluate.py

REM app/ files
type nul > app\streamlit_app.py

REM notebooks/ files
type nul > notebooks\exploration.ipynb

REM root files
type nul > .env
type nul > .gitignore
type nul > requirements.txt
type nul > README.md
type nul > architecture_diagram.png

echo Empty project structure created.
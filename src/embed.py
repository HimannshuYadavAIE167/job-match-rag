import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROCESSED_FILE = Path("data/processed/chunks.json")
VECTOR_STORE_DIR = Path("vector_store")
COLLECTION_NAME = "job_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"


class VectorIndexManager:
    def __init__(
        self,
        persist_dir: Path = VECTOR_STORE_DIR,
        collection_name: str = COLLECTION_NAME,
        model_name: str = MODEL_NAME
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading embedding model: {model_name}...")
        self.embedder = SentenceTransformer(model_name)

        logger.info(f"Initializing ChromaDB client at: {self.persist_dir}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: list[dict[str, Any]], batch_size: int = 64) -> None:
        if not chunks:
            logger.warning("No chunks provided to index.")
            return

        total = len(chunks)
        logger.info(f"Indexing {total} chunks into collection '{self.collection_name}'...")

        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]

            ids = [item["chunk_id"] for item in batch]
            documents = [item["text"] for item in batch]
            metadatas = [item["metadata"] for item in batch]

            # Generate dense vector representations
            embeddings = self.embedder.encode(
                documents,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Processed batch {i + len(batch)}/{total}")

        logger.info(f"Successfully indexed all {total} chunks into ChromaDB.")


def run_indexing() -> None:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Processed chunks file not found at {PROCESSED_FILE}. Run src/preprocess.py first."
        )

    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    indexer = VectorIndexManager()
    indexer.index_chunks(chunks)


if __name__ == "__main__":
    run_indexing()
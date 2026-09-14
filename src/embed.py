"""
Embeds job chunks and indexes them into a local, persistent ChromaDB collection.
"""
import json
import logging
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    PROCESSED_DATA_PATH,
    VECTOR_STORE_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VectorIndexManager:
    def __init__(
        self,
        persist_dir=VECTOR_STORE_DIR,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading embedding model: {model_name}...")
        self.embedder = SentenceTransformer(model_name)

        logger.info(f"Initializing ChromaDB client at: {self.persist_dir}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(self, chunks: list[dict[str, Any]], batch_size: int = 64) -> None:
        if not chunks:
            logger.warning("No chunks provided to index.")
            return

        total = len(chunks)
        logger.info(f"Indexing {total} chunks into collection '{self.collection_name}'...")

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]

            ids = [item["chunk_id"] for item in batch]
            documents = [item["text"] for item in batch]
            metadatas = [item["metadata"] for item in batch]

            embeddings = self.embedder.encode(
                documents,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(f"Processed batch {i + len(batch)}/{total}")

        logger.info(f"Successfully indexed all {total} chunks into ChromaDB.")


def run_indexing() -> None:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed chunks file not found at {PROCESSED_DATA_PATH}. Run src/preprocess.py first."
        )

    with open(PROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    indexer = VectorIndexManager()
    indexer.index_chunks(chunks)


if __name__ == "__main__":
    run_indexing()

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scrape_jobs import fetch_and_save_jobs
from src.preprocess import process_all_jobs
from src.embed import run_indexing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [BACKGROUND WORKER] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Refresh interval (6 hours = 21600 seconds)
UPDATE_INTERVAL_SECONDS = 6 * 60 * 60


def run_ingestion_cycle():
    logger.info("Initiating scheduled job ingestion cycle...")
    try:
        fetch_and_save_jobs(search_term="Machine Learning Engineer", location="Remote", results_wanted=15)
        process_all_jobs()
        run_indexing()
        logger.info(f"Ingestion cycle completed successfully at {datetime.utcnow().isoformat()} UTC.")
    except Exception as e:
        logger.error(f"Error during scheduled ingestion cycle: {e}")


def main():
    logger.info("Job Background Worker started.")
    # Run once immediately on startup
    run_ingestion_cycle()

    while True:
        logger.info(f"Sleeping for {UPDATE_INTERVAL_SECONDS // 3600} hours until next refresh cycle...")
        time.sleep(UPDATE_INTERVAL_SECONDS)
        run_ingestion_cycle()


if __name__ == "__main__":
    main()
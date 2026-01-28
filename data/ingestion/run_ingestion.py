# data/ingestion/run_ingestion.py

from data.ingestion.fetch_jobs import fetch_raw_jobs
from data.ingestion.normalize_jobs import normalize_jobs

if __name__ == "__main__":
    fetch_raw_jobs()
    normalize_jobs()
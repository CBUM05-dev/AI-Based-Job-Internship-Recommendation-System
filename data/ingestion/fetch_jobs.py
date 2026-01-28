# data/ingestion/fetch_jobs.py
# raw data (API / scraping)

import requests
import json
from utils.logger import get_logger

logger = get_logger("JobFetcher")

API_URL = "https://remotive.com/api/remote-jobs"

def fetch_raw_jobs(): 
    logger.info("Fetching jobs from Remotive Api...")
    response = requests.get(API_URL , timeout=10)
    response.raise_for_status()
    
    # C’est une List[Dict] JSON pure
    jobs = response.json()["jobs"]
    # liste de jobs(job = dict) -> .json() -> Transforme la réponse JSON en objet Python
    
    with open("data/ingestion/raw_jobs.json" , "w" , encoding="utf-8") as f:
        json.dump(jobs , f , ensure_ascii=False,indent=2)
        
    logger.info(f"Fetched {len(jobs)} jobs.")
    
    return jobs
    
    
    
    """
requests : Lib HTTP pour appeler des APIs

timeout=10 : Max 10s avant abandon

raise_for_status() : Stop si erreur HTTP :
200–299 → OK
400–499 → erreur client
500–599 → erreur serveur


    """

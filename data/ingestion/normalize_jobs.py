# data/ingestion/normalize_jobs.py

import json 
import pandas as pd
from utils.logger import get_logger
from bs4 import BeautifulSoup


logger = get_logger("JobNormalizer")

def infer_level(text):
    text = text.lower()
    if "senior" in text or "lead" in text or "engineer" in text:
        return "advanced"
    if "junior" in text or "intern" in text :
        return "beginner"
    return "intermediate"

def infer_mode(job):
    return "remote"       # Remotive = remote by definition , (just to fetch real world related jobs)

def infer_domain(job):
    tags = " ".join(job.get("tags",[])).lower()
    if "ai" in tags or "ml" in tags:
        return "AI"
    if "data" in tags :
        return "Data"
    if "web" in tags or "frontend" in tags or "backend" in tags:
        return "Web"
    return "Other"

def clean_html(text):
    return BeautifulSoup(text , "html.parser").get_text(separator=" ")

def normalize_jobs():
    logger.info("Normalizing raw jobs ...")
    
    with open("data/ingestion/raw_jobs.json", encoding="utf-8") as f:
        raw_jobs = json.load(f)
    # json.load(f) retourne exactement ce qui est dans le fichier JSON.
    # raw_jobs = List[Dict]
        
    normalized = []
    
    for idx , job in enumerate(raw_jobs) :
        description = job.get("description", "")
        cleaned_desc = clean_html(description)

        normalized.append({
            "job_id": idx,
            "title": job.get("title", ""),
            "skills": ",".join(job.get("tags", [])),
            "level": infer_level(job.get("title", "") + " " + description),
            "mode": infer_mode(job),
            "domain": infer_domain(job),
            "description": cleaned_desc
        })

    df = pd.DataFrame(normalized)
    df.to_csv("data/raw/jobs.csv", index=False)

    logger.info(f"Saved {len(df)} normalized jobs to data/raw/jobs.csv")
    

"""
data = json.load(file_pointer)	data = json.loads(json_string)
"""
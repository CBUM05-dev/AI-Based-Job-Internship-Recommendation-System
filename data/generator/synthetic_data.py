# data/generator/synthetic_data.py

import pandas as pd
import random 
from utils.logger import get_logger

logger = get_logger("SyntheticDataGenerator")

random.seed(42)


SKILLS_POOL = [
    "python", "sql", "machine learning", "deep learning",
    "fastapi", "docker", "linux", "git",
    "data analysis", "nlp", "computer vision"
]

DOMAINS = ["AI", "Web", "Data", "DevOps"]
LEVELS = ["beginner", "intermediate", "advanced"]
MODES = ["remote", "on-site", "hybrid"]

def generate_jobs(n=400):
    logger.info(f"Generating {n} synthetic job postings...")
    
    jobs = []
    
    for i in range(n):
        job = {
            "job_id" : i ,
            "title" : f"Job_{i}" ,
            "skills" : random.sample(SKILLS_POOL , k=random.randint(3,6)) ,
            "level" : random.choice(LEVELS) ,
            "mode" : random.choice(MODES) ,
            "domain" : random.choice(DOMAINS)
        }
        jobs.append(job)
        
    jobs_df = pd.DataFrame(jobs)
    logger.info("Synthetic job postings generated.")
    return jobs_df

def generate_users(n=250):
    logger.info(f"Generating {n} synthetic user profiles...")
    
    users = []
    
    for i in range(n):
        user = {
            "user_id": i,
            "skills": random.sample(SKILLS_POOL, k=random.randint(2, 5)),
            "level": random.choice(LEVELS),
            "mode": random.choice(MODES),
            "domain": random.choice(DOMAINS)
        }
        users.append(user)
        
    users_df = pd.DataFrame(users)
    logger.info("Synthetic user profiles generated.")
    return users_df

def generate_interactions(users , jobs , n_interactions=3000) :
    logger.info(f"Generating {n_interactions} interactions")
    interactions = []
    
    for _ in range(n_interactions) :
        user = users.sample(1).iloc[0]
        job = jobs.sample(1).iloc[0]
        
        # Simple Relevance rule (realistic)
        skill_overlap = len(set(user["skills"]) & set(job["skills"]))
        level_match = user["level"] == job["level"]
        
        relevance = 1 if skill_overlap >=2 and level_match else 0
        
        interactions.append({
            "user_id" : user["user_id"],
            "job_id" : job["job_id"] ,
            "relevance" : relevance
        })
        
    return pd.DataFrame(interactions)

if __name__ == "__main__" :
    jobs_df = generate_jobs()
    users_df = generate_users()
    interactions_df = generate_interactions(users_df , jobs_df)
    
    jobs_df.to_csv("data/raw/jobs.csv", index=False)
    users_df.to_csv("data/raw/users.csv", index=False)
    interactions_df.to_csv("data/raw/interactions.csv", index=False)

    logger.info("Synthetic datasets generated successfully")
        
    



"""
    🧠 Why this design is VERY important
✅ Skill overlap logic

Mimics real matching

Not random noise

Learnable by ML

✅ Weak but realistic signal

Some matches are bad

Some are good

This is real life

✅ Interaction dataset

This is what enables:

Precision@K

Recall@K

Ranking evaluation

Without this → no real AI system
"""
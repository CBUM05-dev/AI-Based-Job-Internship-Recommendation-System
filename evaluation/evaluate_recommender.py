# evaluation/test_real_jobs.py

import pandas as pd
from features.feature_builder import FeatureBuilder
from utils.logger import get_logger
from sklearn.metrics.pairwise import cosine_similarity

logger = get_logger("EvaluationPhase")

# --------------------------
# 1️⃣ Load real jobs
# --------------------------
jobs_df = pd.read_csv("data/raw/jobs.csv")
logger.info(f"Loaded {len(jobs_df)} jobs from real data.")

# --------------------------
# 2️⃣ Simulate users (or later extract from CV)
# --------------------------
SIMULATED_USERS = [
    {
        "user_id": 0,
        "skills": ["python", "machine learning", "nlp"],
        "level": "beginner",
        "mode": "remote",
        "domain": "AI"
    },
    {
        "user_id": 1,
        "skills": ["docker", "linux", "fastapi"],
        "level": "intermediate",
        "mode": "remote",
        "domain": "DevOps"
    },
    {
        "user_id": 2,
        "skills": ["data analysis", "sql", "python"],
        "level": "advanced",
        "mode": "remote",
        "domain": "Data"
    }
]
users_df = pd.DataFrame(SIMULATED_USERS)
logger.info(f"Simulated {len(users_df)} users for evaluation.")

# --------------------------
# 3️⃣ Compute embeddings
# --------------------------
fb = FeatureBuilder()
logger.info("Embedding jobs...")
job_vectors = fb.transform_jobs(jobs_df)
logger.info("Embedding users...")
user_vectors = fb.transform_users(users_df)

# --------------------------
# 4️⃣ Compute top-5 recommendations for each user
# --------------------------
TOP_K = 5

for i, user in users_df.iterrows():
    logger.info(f"\n=== Top-{TOP_K} recommendations for User {user['user_id']} ===")
    
    # Compute cosine similarity
    sims = cosine_similarity(user_vectors[i].reshape(1, -1), job_vectors)[0]
    
    # Get top indices
    top_indices = sims.argsort()[::-1][:TOP_K]
    top_jobs = jobs_df.iloc[top_indices].copy()
    top_jobs["score"] = sims[top_indices]  # attach similarity scores
    
    # Print nicely
    print(top_jobs[["title", "level", "mode", "domain", "skills", "score"]])
    
    # Optional: summary stats for this user
    print(f"Score range: min={sims.min():.3f}, max={sims.max():.3f}, mean={sims.mean():.3f}")

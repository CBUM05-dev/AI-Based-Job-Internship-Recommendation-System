import pandas as pd
from features.feature_builder import FeatureBuilder
from models.baseline_recommender import BaseLineRecommender
from llm.groq_client import GroqClient
from schemas.request import RecommendationRequest
from utils.logger import get_logger
import json
from api.service import RecommendationService

logger = get_logger("CLI")

client = GroqClient()

# Load Data
jobs = pd.read_csv("data/raw/jobs.csv")
users = pd.read_csv("data/raw/users.csv")


rec_service = RecommendationService(jobs)


print("\n=== AI Job Recommendation CLI ===")
print("Type 'exit' to quit\n")


while True : 
    query_txt = input("Describe what you want (or structured input):\n> ")
    
    if query_txt.lower() == "exit":
        break
    
    # Create a new request object
    req = RecommendationRequest(
        skills=[],
        level="beginner",
        mode="remote"
    )
    
    
    # Parse LLM query
    parsed_str = rec_service.gq_client.parse_user_query(query_txt)
    parsed = json.loads(parsed_str)

    logger.info(f"Parsed user query: {parsed}")

    # Fill the request
    req.skills = parsed.get("skills", [])
    req.level = parsed.get("level", "beginner")
    req.mode = parsed.get("mode", "remote")
    req.domain = parsed.get("domain")
    req.query = query_txt

    logger.info("Extracted request from parsed input successfully!")

    # Get recommendations
    recommendations = rec_service.recommend(req)

    print("\n=== Top Job Recommendations ===")
    print(recommendations[["job_id", "title", "score"]])
    print("\n-----------------------------\n")    
    
    


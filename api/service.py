from features.feature_builder import FeatureBuilder
from llm.groq_client import GroqClient
from models.baseline_recommender import BaseLineRecommender
from utils.logger import get_logger
import json
import pandas as pd
from schemas.request import RecommendationRequest

logger = get_logger("RecommendationService")

class RecommendationService:
    def __init__(self , jobs_df):
        self.gq_client = GroqClient()
        self.fb = FeatureBuilder()
        self.job_embeddings = self.fb.transform_jobs(jobs_df)
        self.recommender = BaseLineRecommender(self.job_embeddings, jobs_df)
        
    def recommend(self , request : RecommendationRequest) :
        logger.info("Starting recommendation process...")
        
        # If user used natural language query
        if request.query :
            logger.info("Natural language query detected. Parsing...")
            parsed =  self.gq_client.parse_user_query(request.query)
            # Convert parsed JSON string to dictionary
            logger.info(f"Parsed user query: {parsed}")
            parsed = json.loads(parsed)
            logger.info(f"Parsed user query to JSON: {parsed}")
            
            user_df = {
                "skills" : ",".join(parsed.get("skills" , [])),
                "level" : parsed.get("level") or "beginner",
                "mode" : parsed.get("mode") or "remote",
                "domain" : parsed.get("domain") or "other"
            }
        else : 
            logger.info("Structured input detected.")
            # if structured input is provided or after parsing
            user_df = {
                "skills" : ",".join(request.skills),
                "level" : request.level,
                "mode" : request.mode,
                "domain" : request.domain
            }
            
        user_vector = self.fb.transform_users(
            pd.DataFrame([user_df])
        )[0]  # Get the first (and only) row as numpy array , we need 1D array
        
        return self.recommender.recommend(user_vector)
            
    
        
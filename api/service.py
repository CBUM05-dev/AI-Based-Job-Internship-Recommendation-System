from features.feature_builder import FeatureBuilder
from llm.groq_client import GroqClient
from models.baseline_recommender import BaseLineRecommender
from utils.logger import get_logger
import json
import pandas as pd
from schemas.request import RecommendationRequest
from llm.parse_utils import safe_parse_llm_output
from parsing.cv_extractor import extract_text_from_pdf

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
            raw_output =  self.gq_client.parse_user_query(request.query)
            # Convert parsed JSON string to dictionary
            logger.info(f"Parsed raw user query: {raw_output}")
            
            parsed = safe_parse_llm_output(raw_output)
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
    # returns a Sortd and get top_k recommendations from the jobs_df
    
    
    # CV-based recommend

    def recommend_from_cv(self, pdf_file):

        text = extract_text_from_pdf(pdf_file)

        raw = self.gq_client.parse_user_query(text)
        parsed = safe_parse_llm_output(raw)
        logger.info(f"Parsed CV profile: {parsed}")

        logger.info(f"Parsed CV profile: {parsed}")

        return self.recommend(
            RecommendationRequest(**parsed)
        )
    
    
    
    
            
    
    
    
        """
✅ Add a safe parser + fallback :
Right now, your LLM returns raw text JSON.
You should never trust LLM output blindly.
        """
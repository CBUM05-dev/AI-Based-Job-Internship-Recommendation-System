# llm/explainer.py
from schemas.request import RecommendationRequest
from utils.logger import get_logger
from llm.parse_utils import safe_parse_llm_output

logger = get_logger("Explainer logger")

class JobExplainer:
    def __init__(self,client):
        self.client = client
    
    def get_user_profile(self , request : RecommendationRequest):
        logger.info("Starting recommendation process...")
        
        # If user used natural language query
        if request.query :
            logger.info("Natural language query detected. Parsing...")
            raw_output =  self.client.parse_user_query(request.query)
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
        return user_df
        
    def explain(self , user_profile , job):
        
        prompt = f"""
Explain why this job matches this candidate.

Candidate:
{user_profile}

Job:
Title: {job['title']}
Skills: {job['skills']}
Description: {job['description']}

Explain in 2–3 concise sentences.
Only use the provided information.
"""
        return self.client.generate(prompt)


        
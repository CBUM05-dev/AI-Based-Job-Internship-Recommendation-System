# llm/groq_client.py
# Groq client for interacting with Groq LLM API

import os
from groq import Groq
from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
logger = get_logger("GroqClient")



class GroqClient:
    def __init__(self):
        Gq_api_key = os.getenv("API_GROQ")
        Gq_model_name = os.getenv("MODEL_GROQ" , "llama-3.1-8b-instant")

        if not Gq_api_key :
            logger.error("Groq API key not found in environment variables.")
            raise ValueError("Groq API key needed.")
        
        client = Groq(api_key=Gq_api_key)
        self.client = client
        self.model = Gq_model_name
        
        logger.info("Groq Client initialized successfully.")
    
    def parse_user_query(self , query : str):
        logger.info("Parsing user query using Groq LLM.")
        
        prompt = f"""
Extract structured information from this request:

"{query}"

Return **strict JSON only, no extra text** with:
- skills (list) : ["skill1", "skill2", ...]
- level : (beginner/intermediate/advanced)
- mode : (remote/on-site/hybrid)
- domain : (e.g., software development, data science, marketing, etc.)
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that helps to parse user queries for job and internship recommendations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0,    
        )
        
        logger.info("User query parsed successfully.")
        return response.choices[0].message.content
    
    def generate(self , prompt:str) -> str:
        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role" : "system" , "content" : "You are an AI assistant that helps to explain a system's result for job and internship recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
        

        

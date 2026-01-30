from fastapi import FastAPI , UploadFile , File
from api.service import RecommendationService
from schemas.request import RecommendationRequest
import pandas as pd
from llm.groq_client import GroqClient
from llm.parse_utils import safe_parse_llm_output
from utils.logger import get_logger
from parsing.cv_extractor import extract_text_from_pdf
from llm.explainer import JobExplainer


logger = get_logger("MainApp")

app = FastAPI(title="AI-Based Job Recommendation System")

jobs = pd.read_csv("data/raw/jobs.csv")

service = RecommendationService(jobs)

gqClient = GroqClient()

# Text-based recommend
@app.post("/recommend")
def recommend(request: RecommendationRequest):
	results = service.recommend(request)
	return results.to_dict(orient="records")


# CV-based recommend
@app.post("/recommend/cv")
async def recommend_from_cv(file : UploadFile = File(...)):
    
    logger.info("CV Uploaded")
    # # returns a Sorted top_k recommendations from the jobs_df
    results = service.recommend_from_cv(file.file)
    return results.to_dict(orient="records")
    
    
    
# Explain recommendtaion results
explainer = JobExplainer(gqClient)
@app.post("/explain_result")
def explain(job_id: int, request: RecommendationRequest):
    user_dict = explainer.get_user_profile(request)
    job = jobs[jobs["job_id"] == job_id].iloc[0].to_dict()
    
    explanation = explainer.explain(
        user_dict,
        job
    )
    return {"explanation" : explanation}

    
    
    
    



"""
df.to_dict(orient="records") method in the pandas library for Python is used to convert a DataFrame into a list of dictionaries,
where each dictionary represents a single row of the DataFrame. 
 
UploadFile → type for uploaded files (CV PDF here)
File(...) → tells FastAPI this is a file input

👉 file: UploadFile = File(...)
Means:
Expect a file from user
Required field (... = mandatory)
Comes from form-data upload
	"""
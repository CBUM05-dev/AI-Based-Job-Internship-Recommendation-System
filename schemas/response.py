# schemas/response.py
# Response schema for job recommendations
# Define API Schemas using Pydantic

from typing import List
from pydantic import BaseModel

# This represents ONE recommendation. A single recommendation item
class JobRecommendation(BaseModel):
    job_id : int
    title : str
    score : float
    
# This represents THE RESPONSE. A response containing many items
# This is semantic modeling, not redundancy.
# We can recommend more than one job, so we need a list of JobRecommendation
class RecommendationResponse(BaseModel):
    recommendations : List[JobRecommendation]


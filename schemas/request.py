# schema/request.py
# Request schema for job recommendations
# Define API Schemas using Pydantic

from pydantic import BaseModel
from typing import List , Optional

class RecommendationRequest(BaseModel):
    skills : List[str] 
    level : str
    mode : str
    domain : Optional[str] = None
    query : Optional[str] = None  # Natural language input for job search
    
# models/baseline_recommender.py
# for the baseline recommender model we choose Cosine Similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  
from utils.logger import get_logger

logger = get_logger("BaselineRecommender")

class BaseLineRecommender:
    def __init__(self , job_embeddings , job_metadata):
        """
        job_embeddings: np.ndarray shape (N, 384) of job embeddings
        job_metadata: original jobs dataframe (for titles, ids, etc.)
        """
        self.job_embeddings = job_embeddings
        self.job_metadata = job_metadata.reset_index(drop=True)
        logger.info("BaselineRecommender initialized.")
        
    def recommend(self , user_vector , top_k=3):
        """
        user_vector: embedding
        """
        logger.info("Generating recommendations...")
        
        # Compute cosine similarity between user vector and all job vectors
        similarities = cosine_similarity(
            [user_vector] ,
            self.job_embeddings     
        )[0]
        
        # Attach similarity scores to job metadata
        results = self.job_metadata.copy()
        results["score"] = similarities
        
        # Sort and get top_k recommendations
        top_results = results.sort_values(
            by = "score" , 
            ascending = False
        ).head(top_k)
        
        logger.info("Recommendations generated.")
        return top_results
        
        
            
            
            

        """
From scikit-learn:

cosine_similarity(X, Y)

It expects:

X: shape (n_samples_X, n_features)

Y: shape (n_samples_Y, n_features)

⚠️ Always 2D arrays, never 1D.

_______________________________
similarities : (1, n_jobs)
Because:
1 user
compared to n jobs
example : array([[0.82, 0.31, 0.67, 0.91]])
)[0]  :  Because the result is 2D, but we only want the row.


        """
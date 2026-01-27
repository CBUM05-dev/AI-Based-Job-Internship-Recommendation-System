# features/feature_builder.py
# New FeatureBuilder (Embedding-Based)
# 👉 Sentence Transformers = des modèles d’EMBEDDINGS


import pandas as pd
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger

logger = get_logger("FeatureBuilder")

embedding_model = "all-MiniLM-L6-v2"

class FeatureBuilder :
    def __init__(self , model_name = embedding_model):
        """
        Embedding-based Feature Builder (open-world)
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Model loaded succesfully")
        
    # ------------------------
    # Text builders
    # ------------------------
    def user_to_text(self , user: dict) -> str:
        return f"""
        Skills: {", ".join(user.get('skills' , []))}
        Level: {user.get('level' , '')}
        Mode: {user.get('mode' , '')} 
        Domain: {user.get('domain','')}
        """.strip()
        
    def job_to_text(self, job: dict) -> str:
        return f"""
        Title: {job.get('title' , '')}
        Skills: {job.get('skills' , '')}
        Level: {job.get('level' , '')}
        Mode: {job.get('mode', '')}
        Domain: {job.get('domain' , '')}
        """.strip()    
        
    # ------------------------
    # Embedding methods
    # ------------------------
    def transform_users(self , users_df: pd.DataFrame):
        logger.info("Embedding users")
        texts = users_df.apply(
            lambda row : self.user_to_text(row.to_dict()) , 1
        ).tolist()
        
        embeddings = self.model.encode(texts , show_progress_bar=True)
        return embeddings
    
    def transform_jobs(self , jobs_df : pd.DataFrame):
        logger.info("Embedding jobs")
        texts = jobs_df.apply(
            lambda row: self.job_to_text(row.to_dict()) , axis=1
        ).tolist()
        
        embeddings = self.model.encode(texts , show_progress_bar = True , normalize_embeddings=True)
        return embeddings
        


        """
Hybrid Recommender :
So you KEEP BOTH:

🔹 Embeddings → semantic match

🔹 Level / mode / domain → control & explainability
        
🔑 Point CRUCIAL :
Le modèle impose la taille du vecteur
384 = dépend du modèle (all-MiniLM-L6-v2)
User embedding → 384 dims

Job embedding → 384 dims
✔️ toujours comparables

5️⃣ model.encode(texts)

Input : List[str] ou str

Output : np.ndarray shape (N, 384)

💥 Parfait pour cosine similarity


⚠️ Petit détail à améliorer (niveau ingénieur)

Ajoute ça pour être clean :

embeddings = self.model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)


👉 Ça te permet :

cosine similarity = simple dot product

plus stable numériquement

🔑 Point clé (à graver)

normalize_embeddings=True
❌ ne change PAS la dimension
✅ change seulement la longueur (norme) du vecteur

👉 384 paramètres restent 384 paramètres

🔹 Avec normalize_embeddings=True

Le modèle fait :
v_normalized = v / ||v||

Résultat :
||v_normalized|| = 1

❓ Pourquoi on fait ça ?
Cosine similarity (définition)
cos(u, v) = (u · v) / (||u|| * ||v||)

MAIS si ||u|| = 1 et ||v|| = 1 :
cos(u, v) = u · v


💥 Le dot product devient EXACTEMENT la cosine similarity 
Pourquoi	vitesse + stabilité
        """
 # features/feature_builder.py

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder , OrdinalEncoder

class FeatureBuilder :
    def __init__(self):
        # One-hot encoder for skills
        self.skill_encoder = MultiLabelBinarizer()
        # Ordinal encoder for experience levels (order matters)
        self.level_encoder = OrdinalEncoder()
        # Label encoder for domain (no order)
        self.domain_encoder = LabelEncoder()
        # Mode mapping for job type
        self.mode_mapping = {"remote":1 , "onsite":0}
        
    def fit(self , jobs_df , users_df):
        # Fit skill encoder on all skills
        all_skills = jobs_df["skills"].apply(lambda x : x.split(",")).tolist()
        all_user_skills = users_df["skills"].apply(lambda x : x.split(',')).tolist()
        # Fited on the combined list of skills from jobs and users
        # Iterable of Iterables , each inner iterable is a list of skills
        self.skill_encoder.fit(all_skills + all_user_skills)
        
        # Fit level encoder 
        experience_levels  = jobs_df[["level"]]
        self.level_encoder.fit(experience_levels)
        
        # Fit domain encoder
        domains = pd.concat([jobs_df["domain"], users_df['domain']]).fillna("other")        
        # Add "other" if not in the data
        if "other" not in domains.values:
            domains = pd.concat([domains, pd.Series(["other"])])
        self.domain_encoder.fit(domains)
        
    def transform_jobs(self , jobs_df):
        
        jobs_df = jobs_df.copy()
        # Skills Vector
        # MultiLabelBinarizer: Input is a Series of lists (Iterable of Iterables)
        # No .tolist() needed, but consistent with fit ()
        # .apply(...). This created a Pandas Series of Lists.worked because Both are "iterables of iterables."
        skills = self.skill_encoder.transform(jobs_df["skills"].apply(lambda x : x.split(",")))
        skills_df = pd.DataFrame(skills , columns = self.skill_encoder.classes_ , index = jobs_df.index)
        
        # Level numeric :
        # OrdinalEncoder expects 2D array-like, so we pass a DataFrame
        level = self.level_encoder.transform(jobs_df[["level"]]).flatten()
        jobs_df["level_num"] = level
        
        # Mode numeric :
        jobs_df["mode_num"] = jobs_df["mode"].map(self.mode_mapping)        
        
        # Domain numeric :
        # Label Encoder accepts 1D array-like, so we pass a Series directly
        domain = self.domain_encoder.transform(jobs_df["domain"])
        jobs_df["domain_num"] = domain  
        
        # Combine all features
        job_features = pd.concat([skills_df , jobs_df[["level_num", "mode_num", "domain_num"]]], axis=1)

        return job_features
    
    def safe_transform_level(self, level_series):
        # If level is unknown, map to "beginner" (or lowest level)
        known_levels = self.level_encoder.categories_[0].tolist()
        return level_series.apply(lambda x: x if x in known_levels else "beginner")
    
    def safe_transform_domain(self, domain_series):
        known_domains = self.domain_encoder.classes_.tolist()
        return domain_series.apply(lambda x: x if x in known_domains else "other")
    
    def transform_users(self , users_df) :
        users_df = users_df.copy()
        
        # Skills Vector
        skills = self.skill_encoder.transform(users_df["skills"].apply(lambda x : x.split(",")))
        skills_df = pd.DataFrame(skills , columns = self.skill_encoder.classes_ , index = users_df.index)
        
        # Level numeric :
        users_df["level"] = self.safe_transform_level(users_df["level"])
        level = self.level_encoder.transform(users_df[["level"]]).flatten()
        users_df["level_num"] = level
        
        # mode numeric :
        users_df["mode"] = users_df["mode"].fillna("remote")  # fallback
        users_df["mode_num"] = users_df["mode"].apply(lambda x: self.mode_mapping.get(str(x).lower(), 1))  # default remote
        
        # Domain numeric :
        users_df["domain"] = users_df["domain"].fillna("other")  # fallback
        users_df["domain"] = self.safe_transform_domain(users_df["domain"])
        domain = self.domain_encoder.transform(users_df["domain"])
        users_df["domain_num"] = domain  
        
        # Combine all features
        user_features = pd.concat([skills_df , users_df[['level_num', 'mode_num', 'domain_num']]], axis=1)

        return user_features
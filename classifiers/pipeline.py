import os
import joblib
import math
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

import lightgbm as lgb
from preprocessor import Preprocessor

from Clf_phishing_cnn import Clf_phishing_cnn
from Clf_malware_cnn import Clf_malware_cnn
from Clf_phishing_lgbm import Clf_phishing_lgbm

class Pipeline:
    def __init__(self):
        """
        Initializes the classification pipeline.
        """

        # Initialize paths
        self.module_dir = os.path.dirname(__file__)
        #self.model_path = os.path.join(self.module_dir, "models")


        # Initialize preprocessor
        self.pp = Preprocessor()

        # Load classifiers
        self.clf_phishing_cnn = Clf_phishing_cnn()
        self.clf_phishing_lgbm = Clf_phishing_lgbm()
        self.clf_malware_cnn = Clf_malware_cnn()


    def classify_domain(self, domain_name: str, feature_vector: dict) -> dict:
        """
        Classifies the domain using the trained models and returns the results.
        """
        result = {
            "domain": domain_name,    
            "aggregate_probability": 0.7898383552053838,
            "aggregate_description": "...",
            "classification_results": [
                {
                    "classifier": "Phishing",
                    "probability": 0.05633430716219098,
                    "description": "No phishing detected."
                },
                {
                    "classifier": "Malware",
                    "probability": 0.004824631535984588,
                    "description": "No malware detected."
                },
                {
                    "classifier": "DGA",
                    "probability": 0.8888312407957214,
                    "description": "The domain has high level of DGA incidators.",
                    "details": {
                        "dga:fit": "80.00%",
                        "dga:vut": "20.00%"
                    }
                }
            ]
        }

        return result
    

    def feature_statistics(self, domain_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates feature statistics for the domain data
        """
        # Define the prefixes
        prefixes = ['dns_', 'tls_', 'ip_', 'rdap_', 'geo_'] # lex_ is always present

        # Initialize a DataFrame with domain names only
        
        stats = domain_data[['domain_name']].copy()
        #stats.set_index('domain_name', inplace=True)

        # Iterate through each prefix to calculate the required ratios
        for prefix in prefixes:
            # Filter columns with the current prefix
            prefixed_columns = [col for col in domain_data.columns if col.startswith(prefix)]
            
            # Calculate the availability ratio (non-None values)
            stats[f'{prefix}available'] = domain_data[prefixed_columns].notna().mean(axis=1)
            
            # Calculate the nonzero ratio (non-zero values, treating None as zero)
            stats[f'{prefix}nonzero'] = domain_data[prefixed_columns].fillna(0).astype(bool).mean(axis=1)
        
        return stats
    

    def calculate_badness_probability(self, domain: pd.Series) -> float:
        """
        Calculates the badness probability based on the results of invividual classifiers
        and statistical properties of the domain features.
        """
        
        return (domain["phishing_cnn_result"] + domain["malware_cnn_result"]) / 2  # Just for testing
    

    def generate_result(self, stats: pd.Series) -> dict:
        """
        Generated the final classification result for a single domain name
        """
        result = {
            "domain": stats["domain_name"],
            "aggregate_probability": stats["badness_probability"],
            "aggregate_description": "...",
            "classification_results": [
                {
                    "classifier": "Phishing",
                    "probability": stats["phishing_cnn_result"],
                    "description": "aaa.",
                    "details": {
                        "CNN phishing classifier": stats["phishing_cnn_result"],
                        "LightGBM phishing classifier": stats["phishing_lgbm_result"],
                    }
                },
                {
                    "classifier": "Malware",
                    "probability": stats["malware_cnn_result"],
                    "description": "aaa.",
                    "details:": {
                        "CNN malware classifier": stats["malware_cnn_result"]
                    }
                },
            ]
        }
        return result


    def classify_domains(self, df: pd.DataFrame) -> list[dict]:
        """
        Classifies the domains from a pandas df and returns list the results.
        Each row of the input DF is a single domain, represented by a column  domain_name
        and 
        """
        # The domain name should be the index
        #df.set_index('domain_name', inplace=True)
        
        # Calculate the feature statistics
        stats = self.feature_statistics(df)
        
        # Get NDF representation of the data for each classifier
        ndf_phishing = self.pp.NDF(df, "phishing")
        ndf_malware = self.pp.NDF(df, "malware")
        ndf_dga_binary = self.pp.NDF(df, "dga_binary")
        ndf_dga_multiclass = self.pp.NDF(df, "dga_multiclass")
    
        # Get individual classifiers' results
        stats["phishing_cnn_result"] = self.clf_phishing_cnn.classify(ndf_phishing)
        stats["phishing_lgbm_result"] = self.clf_phishing_lgbm.classify(ndf_phishing)
        stats["malware_cnn_result"] = self.clf_phishing_cnn.classify(ndf_malware)

        # Calculate the overall badness probability
        stats["badness_probability"] = stats.apply(self.calculate_badness_probability, axis=1)

        # Create an array of results
        results = stats.apply(lambda domain_stats: self.generate_result(domain_stats), axis=1).tolist()

        return results


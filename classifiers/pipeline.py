import os
import pickle 
import pandas as pd

import lightgbm as lgb

class Pipeline:
    def __init__(self):
        """
        Initializes the classification pipeline.
        """

        # Initialize paths
        self.module_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(self.module_dir, "models")

        # Load ML models
        self.pishing_model = self.phishing_model = self.loadModel("phishing-lgbm.pkl")


    def loadModel(self, model_name: str) -> object:
        """
        Loads the model from the model directory.
        """
        model_path = os.path.join(self.model_path, model_name)
        model = pickle.load(open(model_path, "rb"))
        return model


    def classifyDomain(self, domain_name: str, feature_vector: dict) -> dict:
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
        prefixes = ['lex_', 'dns_', 'tls_', 'ip_', 'rdap_', 'geo_']

        # Initialize a DataFrame with domain names only
        stats = domain_data[['domain_name']].copy()
        stats.set_index('domain_name', inplace=True)

        # Iterate through each prefix to calculate the required ratios
        for prefix in prefixes:
            # Filter columns with the current prefix
            prefixed_columns = [col for col in domain_data.columns if col.startswith(prefix)]
            
            # Calculate the availability ratio (non-None values)
            stats[f'{prefix}available'] = domain_data[prefixed_columns].notna().mean(axis=1)
            
            # Calculate the nonzero ratio (non-zero values, treating None as zero)
            stats[f'{prefix}nonzero'] = domain_data[prefixed_columns].fillna(0).astype(bool).mean(axis=1)
        
        return stats


    def classifyDomains(self, df: pd.DataFrame) -> list[dict]:
        """
        Classifies the domains from a pandas df and returns list the results.
        Each row of the input DF is a single domain, represented by a column  domain_name
        and 
        """
        # The domain name should be the index
        df.set_index('domain_name', inplace=True)
        
        # Calculate the feature statistics
        stats = self.calculate_availability(df)
        
        # Get NDF representation of the data for each classifier
        ndf_phishing = self.pp.NDF(df, "phishing")
        ndf_malware = self.pp.NDF(df, "malware")
        ndf_dga = self.pp.NDF(df, "dga")

        # TODO

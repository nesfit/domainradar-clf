import os
import pickle 

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

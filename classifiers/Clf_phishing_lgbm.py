"""
Phishing LightGBM classifier for DomainRadar

Classifies phishing domains using the Light Gradient-Boosting Machine (LightGBM) model.
"""
__author__ = "Radek Hranicky"

import joblib
import lightgbm as lgb
import numpy as np

class Clf_phishing_lgbm:
    """
        Class for the LightGBM phishing classifier.
        Expects the model loaded in the ./models/ directory.
        Use the `classify` method to classify a dataset of domain names.
    """

    def __init__(self):
        """
        Initializes the classifier.
        """

        # Load the LightGBM model
        self.model = joblib.load('models/phishing_lgbm_model.joblib')

        # Get the number of features expected by the model
        self.expected_feature_size = self.model.n_features_


    def classify(self, ndf_data: dict) -> np.array:
        """
        Classifies phishing domain names using the LightGBM model.
        The input is an NDF representation of feature vectors, one for each domain.
        Returns a numpy array of malicious class probabilities.
        """

        # Extract features for prediction
        X = np.array(ndf_data['features'].tolist())

        if X.shape[1] != self.expected_feature_size:
            raise ValueError(f"The input data has {X.shape[1]} features, but the model expects {self.expected_feature_size} features.")

        # Predict probabilities for the positive class
        #probabilities = self.model.predict_proba(X)[:, 1]  # Extract the probability of class 1 (positive class)
        probabilities = self.model.predict_proba(X)[:, 0] 

        return probabilities

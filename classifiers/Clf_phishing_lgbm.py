"""
Phishing LightGBM classifier for DomainRadar

Classifies phishing domains using the Light Gradient-Boosting Machine (LightGBM) model.
"""
__author__ = "Radek Hranicky"

import os

import joblib
import numpy as np
from pandas import DataFrame
from pandas.core.dtypes import common as com


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

        # Get the directory of the current file
        base_dir = os.path.dirname(__file__)

        # Load the LightGBM model
        self.model = joblib.load(os.path.join(base_dir, 'models/phishing_lgbm_model_nonndf.joblib'))

        # Get the number of features expected by the model
        self.expected_feature_size = self.model.n_features_

    
    def cast_timestamp(self, df: DataFrame):
        """
        Cast timestamp fields to seconds since epoch.
        """
        for col in df.columns:
            if com.is_timedelta64_dtype(df[col]):
                df[col] = df[col].dt.total_seconds()  # This converts timedelta to float (seconds)
            elif com.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(np.int64) // 10**9  # Converts datetime64 to Unix timestamp (seconds)

        return df


    def classify_ndf(self, ndf_data: dict) -> np.array:
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
        probabilities = self.model.predict_proba(X)[:, 1]  # Extract the probability of class 1 (positive class)
        #probabilities = self.model.predict_proba(X)[:, 0] 

        return probabilities


    def classify(self, input_data: DataFrame) -> list:
        # Load the trained model
        
        # Drop the 'domain_name' column if it exists
        if 'domain_name' in input_data.columns:
            input_data = input_data.drop('domain_name', axis=1)

        # Drop the 'label' column if it exists
        if 'label' in input_data.columns:
            input_data = input_data.drop('label', axis=1)
        
        # Cast timestamps
        input_data = self.cast_timestamp(input_data)
        
        # Handle NaNs
        input_data.fillna(-1, inplace=True)
        
        # Predict the probabilities of the positive class (malware)
        probabilities = self.model.predict_proba(input_data)[:, 1]
        
        return probabilities
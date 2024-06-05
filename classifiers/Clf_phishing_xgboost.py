"""
Phishing LightGBM classifier for DomainRadar

Classifies phishing domains using the eXtreme Gradient Boosting (XGBoost) model.
"""
__author__ = "Radek Hranicky"

import joblib
import xgboost as lgb
import numpy as np
import os

from pandas import DataFrame
from pandas.core.dtypes import common as com
from classifiers.options import PipelineOptions

class Clf_phishing_xgboost:
    """
        Class for the XGBoost phishing classifier.
        Expects the model loaded in the ./models/ directory.
        Use the `classify` method to classify a dataset of domain names.
    """

    def __init__(self, options: PipelineOptions):
        """
        Initializes the classifier.
        """

        # Get the directory of the current file
        self.base_dir = os.path.dirname(__file__)

        # Load the XGBoost model
        self.model = joblib.load(os.path.join(options.models_dir, 'phishing_xgboost_model_nonndf.joblib'))

        # Get the number of features expected by the model
        self.expected_feature_size = self.model.n_features_in_


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
"""
Phishing LightGBM classifier for DomainRadar

Classifies phishing domains using the TF-IDF method.
"""
__author__ = "Alisher Mazhirinov"

import os
import shap
import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.dtypes import common as com

from classifiers.options import PipelineOptions

class Clf_phishing_tfidf_lgbm:
    """
        Class for the TFIDF-based LightGBM phishing classifier.
        Expects the model loaded in the ./models/ directory.
        Use the `classify` method to classify a dataset of domain names.
    """

    def __init__(self, options: PipelineOptions):
        """
        Initializes the classifier.
        """

        # Load the LightGBM model
        self.model = joblib.load(os.path.join(options.models_dir, 'lgbm_phishing_model.joblib'))
        print("NUM:")
        print(self.model.n_features_)

        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        # Get the number of features expected by the model
        self.expected_feature_size = self.model.n_features_

        # Columns that are not used in the model
        self.disqualified_columns = []



    def debug_domain(self, domain_name: str, feature_vectors: DataFrame, n_top_features: int = 10):
        """
        Debug a specific domain by calculating the feature importance for its classification.
        
        Args:
            domain_name (str): The domain name to debug.
            input_data (DataFrame): The feature vector of the domain.
            n_top_features (int, optional): Number of top features to display. Default is 10.
        """

        input_data = feature_vectors.copy()

        # Only preserve tfidf_ columns
        input_data = input_data[[col for col in input_data.columns if col.startswith('tfidf_phishing_')]]

        # Handle NaNs
        domain_row.fillna(-1, inplace=True)

        # Calculate SHAP values for the domain
        shap_values = self.explainer.shap_values(domain_row)

        # Get the base value (average model output)
        base_value = self.explainer.expected_value

        # Get feature importance for the specific prediction
        domain_shap_values = shap_values[0]  # Since predict_proba is used, shap_values is a list
        domain_feature_importance = zip(domain_row.columns, domain_shap_values)

        # Sort features by absolute SHAP value
        sorted_feature_importance = sorted(domain_feature_importance, key=lambda x: abs(x[1]), reverse=True)

        # Get the top n features
        top_features = sorted_feature_importance[:n_top_features]

        # Store the top features and their values in a dictionary
        feature_info = []
        for feature, importance in top_features:
            feature_info.append({
                "feature": feature,
                "value": domain_row[feature].values[0],
                "shap_value": importance
            })

        # Calculate the probability for the domain
        probability = self.model.predict_proba(domain_row)[:, 1][0]

        # Create data for the force plot
        force_plot_data = (base_value, domain_shap_values, domain_row)

        # Return the information as a dictionary
        return {
            "top_features": feature_info,
            "probability": probability,
            "force_plot_data": force_plot_data
        }
    
    def classify(self, feature_vectors: DataFrame) -> list:

        input_data = feature_vectors.copy()

        print("INPUT DATA:")
        #print(input_data['ti_lang'])

        # Get the names of tfidf_phishing_-prefixed columns
        tfidf_phishing_columns = [col for col in input_data.columns if col.startswith('tfidf_phishing_')]

        # No tfidf data available:  or (input_data['ti_lang'] != 'english').all()
        if not tfidf_phishing_columns:
            # No classification -> return -1 for all
            return [-1] * len(input_data)
        
        print("TFIDF:")
        print(tfidf_phishing_columns)

        # Only preserve tfidf_phishing_ columns
        input_data = input_data[tfidf_phishing_columns]

        # Only preserve tfidf_phishing_ columns
        input_data = input_data[[col for col in input_data.columns if col.startswith('tfidf_phishing_')]]

        # Drop columns that are not used anymore (if they exist)
        """if 'phishing_tfidf_lgbm_result' in input_data.columns:
            input_data.drop(columns=['phishing_tfidf_lgbm_result'], axis=1, inplace=True)
        if 'tfidf_available' in input_data.columns:
            input_data.drop(columns=['tfidf_available'], axis=1, inplace=True)
        if 'tfidf_nonzero' in input_data.columns:
            input_data.drop(columns=['tfidf_nonzero'], axis=1, inplace=True)"""

        # Handle NaNs
        input_data.fillna(-1, inplace=True)

        # Predict the probabilities of the positive class (phishing)
        probabilities = self.model.predict_proba(input_data)[:, 1]

        return probabilities

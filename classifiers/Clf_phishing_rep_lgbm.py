"""
Phishing reputation system classifier for DomainRadar

Classifies phishing domains using LightGBM model based on reputation system features.
"""

__author__ = "Matěj Čech"

import os
import joblib
from pandas import DataFrame

from classifiers.options import PipelineOptions


class Clf_phishing_rep_lgbm:
    """
        Class for the reputation system-based LightGBM phishing classifier.
        Expects the model loaded in the ./models/ directory.
        Use the `classify` method to classify a dataset of domain names.
    """

    def __init__(self, options: PipelineOptions):
        """
        Initializes the classifier.
        """

        # Load the LightGBM model
        self.model = joblib.load(os.path.join(options.models_dir, 'phishing_rep_lgbm_model.joblib'))

        # Get the number of features expected by the model
        self.expected_feature_size = self.model.n_features_

    def classify(self, feature_vectors: DataFrame) -> list:
        input_data = feature_vectors.copy()

        # Get the names of columns that begin with rep_
        rep_columns = [col for col in input_data.columns if col.startswith('rep_')]

        # No rep system data available
        if not rep_columns:
            return [-1] * len(input_data)

        # Only preserve rep_ columns
        input_data = input_data.filter(regex='^rep_')

        # Handle NaNs
        input_data.fillna(-1, inplace=True)

        # Predict the probabilities of the positive class
        probabilities = self.model.predict_proba(input_data)[:, 1]

        return probabilities

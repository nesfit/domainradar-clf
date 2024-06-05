"""
NN-based decision making classifier to predict the overall badness
of the domain name.
"""
__author__ = "Radek Hranicky"

import os
import numpy as np

import joblib
from pandas import DataFrame
from pandas.core.dtypes import common as com
import tensorflow as tf
from tensorflow.keras.models import load_model

from classifiers.options import PipelineOptions

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress most TensorFlow logs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')


class Clf_decision_nn:
    """
        Class for the LightGBM phishing classifier.
        Expects the model loaded in the ./models/ directory.
        Use the `classify` method to classify a dataset of domain names.
    """

    def __init__(self, options: PipelineOptions):
        """
        Initializes the classifier.
        """

        # Load the LightGBM model
        self.model = load_model(os.path.join(options.models_dir, 'decision_nn_model.keras'))

        # Load the scaler
        self.scaler = joblib.load(os.path.join(options.boundaries_dir, 'decision_nn_scaler.joblib'))

        # Get the number of features expected by the model
        # self.expected_feature_size = self.model.n_features_
        self.expected_feature_size = 24

    def cast_timestamp(self, df: DataFrame):
        """
        Cast timestamp fields to seconds since epoch.
        """
        for col in df.columns:
            if com.is_timedelta64_dtype(df[col]):
                df[col] = df[col].dt.total_seconds()  # This converts timedelta to float (seconds)
            elif com.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(np.int64) // 10 ** 9  # Converts datetime64 to Unix timestamp (seconds)

        return df

    def classify(self, input_data: DataFrame) -> list:
        # Load the trained model

        # Drop the domain_name and label columns if exists
        if "domain_name" in input_data.columns:
            input_data.drop(columns=["domain_name"], inplace=True)
        if "label" in input_data.columns:
            input_data.drop(columns=["label"], inplace=True)

        # Check whether the number of features is correct
        if input_data.shape[1] != self.expected_feature_size:
            raise ValueError(
                f"The input data has {input_data.shape[1]} features, but the model expects {self.expected_feature_size} features.")

        # Cast timestamps
        input_data = self.cast_timestamp(input_data)

        # Handle NaNs
        input_data.fillna(-1, inplace=True)

        # Scale the feature matrix using the loaded scaler
        input_data = self.scaler.transform(input_data)

        # Perform predictions
        predictions = self.model.predict(input_data, verbose=0)

        # Extract the probabilities of the positive class (dga)
        positive_class_probabilities = predictions[:, 0]

        return positive_class_probabilities

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

        # Ensure it's a MinMaxScaler
        from sklearn.preprocessing import MinMaxScaler
        if not isinstance(self.scaler, MinMaxScaler):
            raise ValueError("Loaded scaler is not a MinMaxScaler!")

        # NOTE: (not needed - work fine, but only slows)
        # Load feature order from file
        feature_order_file = os.path.join(options.boundaries_dir, "decision_nn_feature_order.txt")
        with open(feature_order_file, "r") as f:
            self.expected_features = [line.strip() for line in f.readlines()]

        # Get the number of features expected by the model
        # self.expected_feature_size = self.model.n_features_
        self.expected_feature_size = 33
        #self.expected_feature_size = 21

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
            
            
        input_data.drop(columns=["malware_tfidf_lgbm_result"], inplace=True)
        input_data.drop(columns=["tfidf_malware_available"], inplace=True)
        input_data.drop(columns=["tfidf_malware_nonzero"], inplace=True)

        input_data.drop(columns=["phishing_tfidf_lgbm_result"], inplace=True)
        input_data.drop(columns=["tfidf_phishing_available"], inplace=True)
        input_data.drop(columns=["tfidf_phishing_nonzero"], inplace=True)


        # NEW: Drop all columns ending in _available or _nonzero
        #cols_to_drop = [col for col in input_data.columns if col.endswith('_available') or col.endswith('_nonzero')]
        #input_data.drop(columns=cols_to_drop, inplace=True)

        # Check whether the number of features is correct
        if input_data.shape[1] != self.expected_feature_size:
            raise ValueError(
                f"The input data has {input_data.shape[1]} features, but the model expects {self.expected_feature_size} features.")


        # NOTE: (not needed - work fine, but only slows)
        # Verify and reorder features if needed
        if set(self.expected_features) != set(input_data.columns):
            raise ValueError("Mismatch between expected features and input features!")
    
        #input_data = input_data[self.expected_features]  # Reorder columns

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

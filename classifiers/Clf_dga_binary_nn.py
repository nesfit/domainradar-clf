"""
Binary DGA/non-DGA classifier using a deep neural network.
"""
__author__ = "Radek Hranicky"


import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import joblib
from pandas import DataFrame
from pandas.core.dtypes import common as com
import tensorflow as tf
from tensorflow.keras.models import load_model

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress most TensorFlow logs
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.config.set_visible_devices([], 'GPU')

class Clf_dga_binary_nn:
    """
        Class for the deep NN DGA binary classifier.
        Expects the model loaded in the ./models/ directory.
        Use the `classify` method to classify a dataset of domain names.
    """

    def __init__(self):
        """
        Initializes the classifier.
        """

        # Get the directory of the current file
        self.base_dir = os.path.dirname(__file__)

        # Load the LightGBM model
        self.model = load_model(os.path.join(self.base_dir, 'models/dga_binary_nn_model.keras'))

        # Load the scaler
        self.scaler = joblib.load(os.path.join(self.base_dir, 'boundaries/dga_binary_nn_scaler.save'))

        # Get the number of features expected by the model
        #self.expected_feature_size = self.model.n_features_

    
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
        
        # Preserve only lex_ columns
        input_data = input_data.filter(regex='^lex_')

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
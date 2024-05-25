# Standard library imports
import os
import datetime
import joblib
import warnings

# Third-party imports for data handling and computation
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import Table
from pandas.core.dtypes import common as com
from pandas import DataFrame

# Machine learning and feature selection libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import torch

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.api.types")
warnings.filterwarnings("ignore", category=UserWarning)


class Preprocessor:
    def __init__(self):
        self.stored = {"dga_binary": dict(), "dga_multiclass": dict(), "phishing": dict(), "malware": dict()}
        self.stored["dga_binary"]["scaler"] = joblib.load("boundaries/dga_binary_scaler.joblib")
        self.stored["dga_binary"]["outliers"] = joblib.load("boundaries/dga_binary_outliers.joblib")
        self.stored["dga_multiclass"]["scaler"] = joblib.load("boundaries/dga_multiclass_scaler.joblib")
        self.stored["dga_multiclass"]["outliers"] = joblib.load("boundaries/dga_multiclass_outliers.joblib")
        self.stored["phishing"]["scaler"] = joblib.load("boundaries/phishing_scaler.joblib")
        self.stored["phishing"]["outliers"] = joblib.load("boundaries/phishing_outliers.joblib")
        self.stored["phishing"]["cf_model"] = joblib.load("models/phishing_ndf_cf_tree.joblib")
        self.stored["malware"]["scaler"] = joblib.load("boundaries/malware_scaler.joblib")
        self.stored["malware"]["outliers"] = joblib.load("boundaries/malware_outliers.joblib")
        self.stored["malware"]["cf_model"] = joblib.load("models/malware_ndf_cf_tree.joblib")

    def apply_scaling(self, df: pd.DataFrame, classifier_type: str):
        numeric_df = df.select_dtypes(include=[np.number])

        # Get the columns that were used during fitting
        fitted_columns = self.stored[classifier_type]["scaler"].feature_names_in_

        # Ensure all fitted columns are present in the numeric_df, filling missing columns with zeros
        for col in fitted_columns:
            if col not in numeric_df.columns:
                numeric_df[col] = 0

        # Transform only the fitted columns (now all columns should be present)
        scaled_data = self.stored[classifier_type]["scaler"].transform(
            numeric_df[fitted_columns]
        )
        scaled_data = 1 / (1 + np.exp(-scaled_data))  # Apply sigmoid scaling

        # Create a DataFrame with the scaled data
        scaled_df = pd.DataFrame(scaled_data, columns=fitted_columns, index=df.index)

        # Add back any non-numeric columns to the DataFrame
        for col in df.columns:
            if col not in numeric_df.columns:
                scaled_df[col] = df[col]

        return scaled_df

    def adjust_outliers(self, features, classifier_type: str):
        # Apply boundaries
        for column, (lower_bound, upper_bound) in self.stored[classifier_type][
            "outliers"
        ].items():
            if (
                column in features.columns
            ):  # Ensure the column exists in the current dataset
                features[column].apply(
                    lambda x: (
                        x
                        if lower_bound <= x <= upper_bound
                        else lower_bound if x < lower_bound else upper_bound
                    )
                )
        return features

    def perform_eda(self, features: pd.DataFrame, classifier_type: str) -> None:
        categorical_features = [
            "lex_tld_hash",
            "geo_continent_hash",
            "geo_countries_hash",
            "rdap_registrar_name_hash",
            "tls_root_authority_hash",
            "tls_leaf_authority_hash",
        ]

        # For non-DGA classifiers, process categorical features with the stored decision tree
        # The result stored as "dtree_prob" then serves as a feature
        # Note: This is mostly used for NN classifiers, but has shown to be useful with
        # tree-based models as well
        if not classifier_type.startswith("dga"):
        
            ## Define a function to predict probability for a single row
            def predict_row_probability(row):
                row_df = (
                    row[categorical_features].to_frame().T
                )  # Ensure the row is a DataFrame
                return self.stored[classifier_type]["cf_model"].predict_proba(row_df)[0, 1]

            # Apply the function to each row and create a new column 'dtree_prob'
            features["dtree_prob"] = features.apply(predict_row_probability, axis=1)

        # Process timestamps
        for col in features.columns:
            if com.is_timedelta64_dtype(features[col]):
                features[col] = features[col].dt.total_seconds()
            elif com.is_datetime64_any_dtype(features[col]):
                features[col] = features[col].astype(np.int64) // 10**9

        # Convert bool columns to float
        for column in features.columns:
            if features[column].dtype == "bool":
                features[column] = features[column].astype("float64")

        # Drop the domain name column
        features = features.drop(features.columns[0], axis=1)

        # Handling missing values in features
        features.fillna(-1, inplace=True)

        # Adjust outliers
        features = self.adjust_outliers(features, classifier_type)

        # Apply scaling
        features = self.apply_scaling(features, classifier_type)

        feature_names = features.columns
        return torch.tensor(features.values).float(), feature_names

    def NDF(self, input_data: pd.DataFrame, classifier_type: str):
        features, feature_names = self.perform_eda(input_data, classifier_type)

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        dataset_name = f"dataset_{current_date}"

        dataset = {
            "name": dataset_name,
            "features": features,
            "labels": [None for _ in range(features.shape[0])],
            "dimension": features.shape[1],
            "feature_names": feature_names,
            "one_line_processing": True,
        }

        return dataset

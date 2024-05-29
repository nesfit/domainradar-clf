"""
Data preprocessor for classifiers

This module contains the Preprocessor class, which is used to preprocess data
for the classifier. Notably, it provides conversion into the NDF format.

Transformations uses stored values of scalers, outliers, and decision trees
that were obtained during the exploratory data analysis (EDA) on the training
dataset. Therefore, this class is intended for use in the production environment
where we classify new data using existing pre-trained models.

IMPORTANT: DO NOT use this class for data processing when training new models!
"""
__authors__ = ["Petr Pouc (invention of NDF, original implementation)",
               "Radek Hranicky (lightweight reimplementation for production)"]

# Standard library imports
import os
import datetime
import joblib
import warnings

# Third-party imports for data handling and computation
import numpy as np
import pandas as pd
from pandas.core.dtypes import common as com

# Machine learning and feature selection libraries
import torch

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.api.types")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


class Preprocessor:
    """
    Class for preprocessing data for the classifiers in a production environment.
    Uses pre-computed boundary and scaler configuration.
    """

    def __init__(self):
        """
        Initializes the preprocessor with the stored values of scalers, outliers, etc.
        """
        # Get the directory of the current file
        base_dir = os.path.dirname(__file__)

        self.stored = {"dga_binary": dict(), "dga_multiclass": dict(), "phishing": dict(), "malware": dict()}
        self.stored["dga_binary"]["scaler"] = joblib.load(os.path.join(base_dir, "boundaries/dga_binary_scaler.joblib"))
        self.stored["dga_binary"]["outliers"] = joblib.load(
            os.path.join(base_dir, "boundaries/dga_binary_outliers.joblib"))
        self.stored["dga_multiclass"]["scaler"] = joblib.load(
            os.path.join(base_dir, "boundaries/dga_multiclass_scaler.joblib"))
        self.stored["dga_multiclass"]["outliers"] = joblib.load(
            os.path.join(base_dir, "boundaries/dga_multiclass_outliers.joblib"))
        self.stored["phishing"]["scaler"] = joblib.load(os.path.join(base_dir, "boundaries/phishing_scaler.joblib"))
        self.stored["phishing"]["outliers"] = joblib.load(os.path.join(base_dir, "boundaries/phishing_outliers.joblib"))
        self.stored["phishing"]["cf_model"] = joblib.load(os.path.join(base_dir, "models/phishing_ndf_cf_tree.joblib"))
        self.stored["malware"]["scaler"] = joblib.load(os.path.join(base_dir, "boundaries/malware_scaler.joblib"))
        self.stored["malware"]["outliers"] = joblib.load(os.path.join(base_dir, "boundaries/malware_outliers.joblib"))
        self.stored["malware"]["cf_model"] = joblib.load(os.path.join(base_dir, "models/malware_ndf_cf_tree.joblib"))

    def apply_scaling(self, df: pd.DataFrame, classifier_type: str):
        """
        Applies scaling to the input DataFrame based on the stored MinMaxScaler.
        The result is then processed using the sigmoid transformation to fit
        into the range between 0 and 1.
        """
        numeric_df = df.select_dtypes(include=[np.number])

        # Get the columns that were used during fitting
        fitted_columns = self.stored[classifier_type]["scaler"].feature_names_in_

        # From them, get those that exist in the current DataFrame
        existing_fitted_columns = [col for col in fitted_columns if col in numeric_df.columns]

        if not existing_fitted_columns:
            raise ValueError("None of the fitted columns are present in the input DataFrame.")

        if len(fitted_columns) != len(numeric_df.columns):
            # Create a DataFrame with zeroes for missing columns (required for proper transformation shape)
            temp_df = pd.DataFrame(0, index=numeric_df.index, columns=fitted_columns, dtype=float)
            # Ensure that the types match by casting numeric_df to float before updating temp_df
            temp_df.update(numeric_df.astype(float))

            # Transform only the fitted columns
            scaled_data = self.stored[classifier_type]["scaler"].transform(temp_df[fitted_columns])
            columns_to_use = fitted_columns
        else:
            # Transform the existing fitted columns directly
            scaled_data = self.stored[classifier_type]["scaler"].transform(numeric_df[existing_fitted_columns])
            columns_to_use = existing_fitted_columns

        scaled_data = 1 / (1 + np.exp(-scaled_data))  # Apply sigmoid scaling

        # Create a DataFrame with the scaled data
        scaled_df = pd.DataFrame(scaled_data, columns=columns_to_use, index=df.index)

        # Filter the scaled_df to keep only the columns that were present in the input DataFrame
        final_scaled_df = scaled_df[existing_fitted_columns]

        # Add back any non-numeric columns to the DataFrame
        for col in df.columns:
            if col not in numeric_df.columns:
                final_scaled_df[col] = df[col]

        return final_scaled_df

    def adjust_outliers(self, features, classifier_type: str):
        """
        Adjusts the outliers in the features based on the stored boundaries.
        """
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

    def apply_eda(self, features: pd.DataFrame, classifier_type: str, drop_categorical=True) -> None:
        """
        Applies feature transformations like scaling, encoding, outlier handling,
        and categorical feature processing using store values obtained
        by exploratory data analysis (EDA) on the training dataset.
        """

        categorical_features = [
            "lex_tld_hash",
            "geo_continent_hash",
            "geo_countries_hash",
            "rdap_registrar_name_hash",
            "tls_root_authority_hash",
            "tls_leaf_authority_hash",
        ]

        if classifier_type.startswith("dga"):
            # For DGA classifiers, preserve only features that start with lex_ (and the domain name)

            columns_to_keep = [col for col in features.columns if col.startswith("lex_")]

            # Keep only the specified columns
            features = features[columns_to_keep]

        else:
            # For non-DGA classifiers, process categorical features with the stored decision tree
            # The result stored as "dtree_prob" then serves as a feature
            # Note: This is mostly used for NN classifiers, but has shown to be useful with
            # tree-based models as well

            ## Define a function to predict probability for a single row
            def predict_row_probability(row):
                row_df = (
                    row[categorical_features].to_frame().T
                )  # Ensure the row is a DataFrame
                return self.stored[classifier_type]["cf_model"].predict_proba(row_df)[0, 1]

            # Apply the function to each row and create a new column 'dtree_prob'
            features["dtree_prob"] = features.apply(predict_row_probability, axis=1)

            if drop_categorical:  # By default True
                # Drop the categorical features
                features.drop(columns=categorical_features, errors='ignore', inplace=True)

        # Drop the domain name column (if present)
        if 'domain_name' in features.columns:
            features = features.drop(columns=['domain_name'])

        # Drop the label column (if present)
        if 'label' in features.columns:
            features = features.drop(columns=['label'])

        # Process timestamps
        for col in features.columns:
            if com.is_timedelta64_dtype(features[col]):
                features[col] = features[col].dt.total_seconds()
            elif com.is_datetime64_any_dtype(features[col]):
                features[col] = features[col].astype(np.int64) // 10 ** 9

        # Convert bool columns to float
        for column in features.columns:
            if features[column].dtype == "bool":
                features[column] = features[column].astype("float64")

        # Handling missing values in features
        features.fillna(-1, inplace=True)

        # Adjust outliers
        features = self.adjust_outliers(features, classifier_type)

        # Apply scaling
        features = self.apply_scaling(features, classifier_type)

        feature_names = features.columns

        return torch.tensor(features.values).float(), feature_names

    def df_to_NDF(self, input_data: pd.DataFrame, classifier_type: str, drop_categorical=True):
        """
        Preprocesses the input data into the NDF format for the classifier.
        This method of NDF creation is intended for the classification
        in the production environment. DO NOT use it for training new models!
        The method expects scalers, outliers, and decision tree for processing
        categorical features to be stored in appropriate paths.
        """

        domain_data = input_data.copy()

        features, feature_names = self.apply_eda(domain_data, classifier_type, drop_categorical=drop_categorical)

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

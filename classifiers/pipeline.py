import importlib.util
import os
import pandas as pd
import warnings

from .preprocessor import Preprocessor

from .Clf_phishing_cnn import Clf_phishing_cnn
from .Clf_malware_cnn import Clf_malware_cnn
from .Clf_phishing_lgbm import Clf_phishing_lgbm
from .Clf_malware_xgboost import Clf_malware_xgboost
from .Clf_dga_binary_nn import Clf_dga_binary_nn
from .Clf_dga_multiclass_lgbm import Clf_dga_multiclass_lgbm
from .Clf_decision_nn import Clf_decision_nn


class Pipeline:
    def __init__(self):
        """
        Initializes the classification pipeline.
        """

        # Initialize paths
        self.module_dir = os.path.dirname(__file__)
        # self.model_path = os.path.join(self.module_dir, "models")

        # Initialize preprocessor
        self.pp = Preprocessor()

        # Load classifiers
        self.clf_phishing_cnn = Clf_phishing_cnn()
        self.clf_phishing_lgbm = Clf_phishing_lgbm()
        self.clf_malware_cnn = Clf_malware_cnn()
        self.clf_malware_xgboost = Clf_malware_xgboost()
        self.clf_dga_binary_nn = Clf_dga_binary_nn()
        self.clf_dga_multiclass_lgbm = Clf_dga_multiclass_lgbm()
        self.clf_decision_nn = Clf_decision_nn()

        # Suppress FutureWarning
        warnings.simplefilter(action='ignore', category=FutureWarning)

    def feature_statistics(self, domain_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates feature statistics for the domain data
        """
        # Define the prefixes
        prefixes = ['dns_', 'tls_', 'ip_', 'rdap_', 'geo_']  # lex_ is always present

        # Initialize a DataFrame with domain names only

        stats = domain_data[['domain_name']].copy()
        # stats.set_index('domain_name', inplace=True)

        # Iterate through each prefix to calculate the required ratios
        for prefix in prefixes:
            # Filter columns with the current prefix
            prefixed_columns = [col for col in domain_data.columns if col.startswith(prefix)]

            # Calculate the availability ratio (non-None values)
            stats[f'{prefix}available'] = domain_data[prefixed_columns].notna().mean(axis=1)

            # Calculate the nonzero ratio (non-zero values, treating None as zero)
            stats[f'{prefix}nonzero'] = domain_data[prefixed_columns].fillna(0).astype(bool).mean(axis=1)

        return stats

    def calculate_badness_probability(self, domain_stats: pd.Series) -> float:
        """
        Calculates the badness probability based on the results of invividual classifiers
        and statistical properties of the domain features.
        """

        return self.clf_decision_nn.classify(pd.DataFrame([domain_stats]))[0]

        # return domain_stats["total_avg"]  # Just for testing

    def generate_result(self, stats: pd.Series) -> dict:
        """
        Generated the final classification result for a single domain name
        """

        # Phishing description
        phishing_desc = ""
        if stats["phishing_avg"] < 0.1:
            phishing_desc = "No phishing detected."
        elif stats["phishing_avg"] >= 0.1 and stats["phishing_avg"] < 0.5:
            phishing_desc = "The domain has some similarities to phishing domains."
        elif stats["phishing_avg"] >= 0.5 and stats["phishing_avg"] < 0.9:
            phishing_desc = "The domain has high level of phishing indicators."
        elif stats["phishing_avg"] >= 0.9:
            phishing_desc = "The domain is most certainly a phishing domain."

        # Malware description
        malware_desc = ""
        if stats["malware_avg"] < 0.1:
            malware_desc = "No malware detected."
        elif stats["malware_avg"] >= 0.1 and stats["malware_avg"] < 0.5:
            malware_desc = "The domain has some similarities to malware domains."
        elif stats["malware_avg"] >= 0.5 and stats["malware_avg"] < 0.9:
            malware_desc = "The domain has high level of malware indicators."
        elif stats["malware_avg"] >= 0.9:
            malware_desc = "The domain is most certainly a malware domain."

        # DGA description
        dga_desc = ""
        if stats["dga_binary_nn_result"] < 0.1:
            dga_desc = "No DGA detected."
        elif stats["dga_binary_nn_result"] >= 0.1 and stats["dga_binary_nn_result"] < 0.5:
            dga_desc = "The domain has some similarities to DGA domains."
        elif stats["dga_binary_nn_result"] >= 0.5 and stats["dga_binary_nn_result"] < 0.9:
            dga_desc = "The domain has high level of DGA incidators."
        elif stats["dga_binary_nn_result"] >= 0.9:
            dga_desc = "The domain is most certainly a DGA domain."

        dga_family_details = dict()

        if stats["dga_binary_nn_result"] >= 0.5:
            for family_name, family_prob in stats["dga_families"].items():
                dga_family_details[family_name] = str(round(family_prob * 100, 2)) + "%"

            # print("+ --------------------------------- +")
            # print(stats["domain_name"])
            # print(dga_family_details)
            # print("+ --------------------------------- +")

        result = {
            "domain": stats["domain_name"],
            "aggregate_probability": stats["badness_probability"],
            "aggregate_description": "...",
            "classification_results": [
                {
                    "classifier": "Phishing",
                    "probability": stats["phishing_avg"],
                    "description": phishing_desc,
                    "details": {
                        "CNN phishing classifier": str(round(stats["phishing_cnn_result"] * 100, 2)) + "%",
                        "LightGBM phishing classifier": str(round(stats["phishing_lgbm_result"] * 100, 2)) + "%",
                    }
                },
                {
                    "classifier": "Malware",
                    "probability": stats["malware_avg"],
                    "description": malware_desc,
                    "details:": {
                        "CNN malware classifier": str(round(stats["malware_cnn_result"] * 100, 2)) + "%",
                        "XGBoost malware classifier": str(round(stats["malware_xgboost_result"] * 100, 2)) + "%",
                    }
                },
                {
                    "classifier": "DGA",
                    "probability": stats["dga_binary_nn_result"],
                    "description": dga_desc,
                    "details:": dga_family_details
                }
            ]
        }
        return result

    def generate_preliminary_results(self, df: pd.DataFrame, output_file: str = None, add_final=False) -> pd.DataFrame:
        """
        This method is used to generate preliminary results for training and testing
        the final aggregation model. The parquet contains domain name, label, feature
        statistics and results of individual classifiers.
        Optionally, the results can be saved to a Parquet file. To use this feature, the "arrow" or "dev"
        optional dependency group must be installed (poetry install --with arrow).
        """

        # Calculate the feature statistics
        stats = self.feature_statistics(df)

        # Add the label to the statistics (if present in the input DataFrame)
        if "label" in df.columns:
            stats["label"] = df["label"]

        # Get NDF representation of the data for each classifier
        ndf_phishing = self.pp.df_to_NDF(df, "phishing")
        # oldndf_phishing = self.pp.df_to_NDF(df, "phishing", drop_categorical=False)

        ndf_malware = self.pp.df_to_NDF(df, "malware")
        # oldndf_malware = self.pp.df_to_NDF(df, "malware", drop_categorical=False)

        ndf_dga_binary = self.pp.df_to_NDF(df, "dga_binary")
        ndf_dga_multiclass = self.pp.df_to_NDF(df, "dga_multiclass")

        # Get individual classifiers' results
        # Phishing
        stats["phishing_cnn_result"] = self.clf_phishing_cnn.classify(ndf_phishing).astype(float)
        # stats["phishing_lgbm_result"] = self.clf_phishing_lgbm.classify(ndf_phishing)
        stats["phishing_lgbm_result"] = self.clf_phishing_lgbm.classify(df)

        # Malware
        stats["malware_cnn_result"] = self.clf_phishing_cnn.classify(ndf_malware).astype(float)
        # stats["malware_xgboost_result"] = self.clf_malware_xgboost.classify(ndf_malware)
        stats["malware_xgboost_result"] = self.clf_malware_xgboost.classify(df)

        # DGA
        stats["dga_binary_nn_result"] = self.clf_dga_binary_nn.classify(df)
        # dga_families = self.clf_dga_multiclass_lgbm.classify(df) # not needed for training decision-maker

        # Calculate derived statistics (additional inputs for the decision making model)
        no_phishing_classifiers = 2
        no_malware_classifiers = 2
        stats["phishing_sum"] = stats["phishing_cnn_result"] + stats["phishing_lgbm_result"]
        stats["phishing_avg"] = stats["phishing_sum"] / no_phishing_classifiers
        stats["phishing_prod"] = stats["phishing_cnn_result"] * stats["phishing_lgbm_result"]
        stats["malware_sum"] = stats["malware_cnn_result"] + stats["malware_xgboost_result"]
        stats["malware_avg"] = stats["malware_sum"] / no_malware_classifiers
        stats["malware_prod"] = stats["malware_cnn_result"] * stats["malware_xgboost_result"]
        stats["total_sum"] = stats["phishing_sum"] + stats["malware_sum"] + stats["dga_binary_nn_result"]
        stats["total_avg"] = stats["total_sum"] / (no_phishing_classifiers + no_malware_classifiers + 1)
        stats["total_prod"] = stats["phishing_prod"] * stats["malware_prod"] * stats["dga_binary_nn_result"]

        if add_final:
            # Calculate the overall badness probability
            stats["badness_probability"] = stats.apply(self.calculate_badness_probability, axis=1)

        # If an output file path is provided, save the DataFrame as a Parquet file
        if output_file:
            if importlib.util.find_spec("pyarrow") is None:
                warnings.warn("The pyarrow library is not installed. Run `poetry install --with dev`.")
                return stats

            import pyarrow.parquet as pq
            import pyarrow as pa

            table = pa.Table.from_pandas(stats)
            pq.write_table(table, output_file)

        return stats

    def classify_domains(self, df: pd.DataFrame) -> list[dict]:
        """
        Classifies the domains from a pandas df and returns list the results.
        Each row of the input DF is a single domain, represented by a column  domain_name
        and 
        """
        # The domain name should be the index
        # df.set_index('domain_name', inplace=True)

        # Calculate the feature statistics
        stats = self.feature_statistics(df)

        # Get NDF representation of the data for each classifier
        ndf_phishing = self.pp.df_to_NDF(df, "phishing")
        oldndf_phishing = self.pp.df_to_NDF(df, "phishing", drop_categorical=False)

        ndf_malware = self.pp.df_to_NDF(df, "malware")
        oldndf_malware = self.pp.df_to_NDF(df, "malware", drop_categorical=False)

        ndf_dga_binary = self.pp.df_to_NDF(df, "dga_binary")
        ndf_dga_multiclass = self.pp.df_to_NDF(df, "dga_multiclass")

        # Get individual classifiers' results
        # Phishing
        stats["phishing_cnn_result"] = self.clf_phishing_cnn.classify(ndf_phishing).astype(float)
        # stats["phishing_lgbm_result"] = self.clf_phishing_lgbm.classify(ndf_phishing)
        stats["phishing_lgbm_result"] = self.clf_phishing_lgbm.classify(df)

        # Malware
        stats["malware_cnn_result"] = self.clf_phishing_cnn.classify(ndf_malware).astype(float)
        # stats["malware_xgboost_result"] = self.clf_malware_xgboost.classify(ndf_malware)
        stats["malware_xgboost_result"] = self.clf_malware_xgboost.classify(df)

        # DGA binary
        stats["dga_binary_nn_result"] = self.clf_dga_binary_nn.classify(df)

        # Calculate derived statistics (additional inputs for the decision making model)
        no_phishing_classifiers = 2
        no_malware_classifiers = 2
        stats["phishing_sum"] = stats["phishing_cnn_result"] + stats["phishing_lgbm_result"]
        stats["phishing_avg"] = stats["phishing_sum"] / no_phishing_classifiers
        stats["phishing_prod"] = stats["phishing_cnn_result"] * stats["phishing_lgbm_result"]
        stats["malware_sum"] = stats["malware_cnn_result"] + stats["malware_xgboost_result"]
        stats["malware_avg"] = stats["malware_sum"] / no_malware_classifiers
        stats["malware_prod"] = stats["malware_cnn_result"] * stats["malware_xgboost_result"]
        stats["total_sum"] = stats["phishing_sum"] + stats["malware_sum"] + stats["dga_binary_nn_result"]
        stats["total_avg"] = stats["total_sum"] / (no_phishing_classifiers + no_malware_classifiers + 1)
        stats["total_prod"] = stats["phishing_prod"] * stats["malware_prod"] * stats["dga_binary_nn_result"]

        # Calculate the overall badness probability
        stats["badness_probability"] = stats.apply(self.calculate_badness_probability, axis=1)

        # DGA Families
        stats["dga_families"] = self.clf_dga_multiclass_lgbm.classify(df)

        # print("=====================================")
        # print(stats)
        # print("=====================================")

        # Create an array of results
        results = stats.apply(lambda domain_stats: self.generate_result(domain_stats), axis=1).tolist()

        return results

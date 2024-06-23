import datetime
import importlib.util
import pandas as pd
import warnings
import joblib

from .options import PipelineOptions
from .preprocessor import Preprocessor
from .feature_definition import features_in_expected_order

from .Clf_phishing_cnn import Clf_phishing_cnn
from .Clf_malware_cnn import Clf_malware_cnn
from .Clf_phishing_deepnn import Clf_phishing_deepnn
from .Clf_phishing_lgbm import Clf_phishing_lgbm
from .Clf_phishing_xgboost import Clf_phishing_xgboost
from .Clf_phishing_dns_nn import Clf_phishing_dns_nn
from .Clf_phishing_rdap_nn import Clf_phishing_rdap_nn
from .Clf_malware_lgbm import Clf_malware_lgbm
from .Clf_malware_xgboost import Clf_malware_xgboost
from .Clf_dga_binary_nn import Clf_dga_binary_nn
from .Clf_dga_multiclass_lgbm import Clf_dga_multiclass_lgbm
from .Clf_decision_nn import Clf_decision_nn


class Pipeline:
    def __init__(self, options: PipelineOptions | None = None):
        """
        Initializes the classification pipeline.
        """

        if options is None:
            options = PipelineOptions()

        # Initialize preprocessor
        self.pp = Preprocessor(options)

        # Load classifiers
        self.clf_phishing_cnn = Clf_phishing_cnn(options)
        self.clf_phishing_deepnn = Clf_phishing_deepnn(options)
        self.clf_phishing_lgbm = Clf_phishing_lgbm(options)
        self.clf_phishing_xgboost = Clf_phishing_xgboost(options)
        self.clf_phishing_dns_nn = Clf_phishing_dns_nn(options)
        self.clf_phishing_rdap_nn = Clf_phishing_rdap_nn(options)
        self.clf_malware_cnn = Clf_malware_cnn(options)
        self.clf_malware_lgbm = Clf_malware_lgbm(options)
        self.clf_malware_xgboost = Clf_malware_xgboost(options)
        self.clf_dga_binary_nn = Clf_dga_binary_nn(options)
        self.clf_dga_multiclass_lgbm = Clf_dga_multiclass_lgbm(options)
        self.clf_decision_nn = Clf_decision_nn(options)

        # Suppress FutureWarning
        warnings.simplefilter(action='ignore', category=FutureWarning)

    def feature_statistics(self, domain_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates feature statistics for the domain data.
        """
        # Define the prefixes
        prefixes = ['dns_', 'tls_', 'ip_', 'rdap_', 'geo_']  # lex_ is always present

        # Initialize a DataFrame with domain names only
        stats = domain_data[['domain_name']].copy()

        # Iterate through each prefix to calculate the required ratios
        for prefix in prefixes:
            # Filter columns with the current prefix
            prefixed_columns = [col for col in domain_data.columns if col.startswith(prefix)]

            if prefixed_columns:
                # Calculate the availability ratio (non-NaN and non -1 values)
                stats[f'{prefix}available'] = domain_data[prefixed_columns].applymap(lambda x: x != -1 and pd.notna(x)).mean(axis=1)

                # Calculate the nonzero ratio (non-zero values, treating NaN and -1 as zero)
                stats[f'{prefix}nonzero'] = domain_data[prefixed_columns].applymap(lambda x: x != 0 and x != -1 and pd.notna(x)).mean(axis=1)
            else:
                # If no columns with the current prefix exist, set ratios to 0
                stats[f'{prefix}available'] = 0
                stats[f'{prefix}nonzero'] = 0

        return stats

    def calculate_badness_probability(self, domain_stats: pd.Series) -> float:
        """
        Calculates the badness probability based on the results of invividual classifiers
        and statistical properties of the domain features.
        """

        badness_probability = self.clf_decision_nn.classify(pd.DataFrame([domain_stats]))[0]

        # Heuristics
        if not (domain_stats["phishing_avg"] > 0.8 and domain_stats["malware_avg"] > 0.8):
            badness_probability -= 0.1
        elif domain_stats["phishing_avg"] > 0.8 and domain_stats["malware_avg"] > 0.8:
            badness_probability += 0.1

        if not (domain_stats["phishing_avg"] > 0.5 and domain_stats["malware_avg"] > 0.5) or \
            (domain_stats["malware_avg"] > 0.5 and domain_stats["dga_binary_nn_result"] > 0.5):
            badness_probability -= 0.1

        if badness_probability < 0.0:
            badness_probability = 0.0
        elif badness_probability > 1.0:
            badness_probability = 1.0

        return badness_probability

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

        timestamp_ms = int(datetime.datetime.now(datetime.UTC).timestamp() * 1e3)

        result = {
            "domain_name": stats["domain_name"],
            "aggregate_probability": stats["badness_probability"],
            "aggregate_description": "...",
            "classification_results": [
                {
                    "classification_date": timestamp_ms,
                    "classifier": "Phishing",
                    "probability": stats["phishing_avg"],
                    "description": phishing_desc,
                    "details": {
                        "CNN phishing classifier": str(round(stats["phishing_cnn_result"] * 100, 2)) + "%",
                        "LightGBM phishing classifier": str(round(stats["phishing_lgbm_result"] * 100, 2)) + "%",
                        "XGBoost phishing classifier": str(round(stats["phishing_xgboost_result"] * 100, 2)) + "%",
                        "Deep NN phishing classifier": str(round(stats["phishing_deepnn_result"] * 100, 2)) + "%",
                        "DNS-based NN phishing classifier": str(round(stats["phishing_dns_nn_result"] * 100, 2)) + "%",
                        "RDAP-based NN phishing classifier": str(round(stats["phishing_rdap_nn_result"] * 100, 2)) + "%",
                    }
                },
                {
                    "classification_date": timestamp_ms,
                    "classifier": "Malware",
                    "probability": stats["malware_avg"],
                    "description": malware_desc,
                    "details": {
                        "CNN malware classifier": str(round(stats["malware_cnn_result"] * 100, 2)) + "%",
                        "LightGBM malware classifier": str(round(stats["malware_xgboost_result"] * 100, 2)) + "%",
                        "XGBoost malware classifier": str(round(stats["malware_xgboost_result"] * 100, 2)) + "%",
                    }
                },
                {
                    "classification_date": timestamp_ms,
                    "classifier": "DGA",
                    "probability": stats["dga_binary_nn_result"],
                    "description": dga_desc,
                    "details": dga_family_details
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
        #stats["phishing_cnn_result"] = self.clf_phishing_cnn.classify(ndf_phishing).astype(float)
        stats["phishing_cnn_result"] = self.clf_phishing_cnn.classify(df)
        # stats["phishing_lgbm_result"] = self.clf_phishing_lgbm.classify(ndf_phishing)
        stats["phishing_lgbm_result"] = self.clf_phishing_lgbm.classify(df)
        stats["phishing_xgboost_result"] = self.clf_phishing_xgboost.classify(df)
        stats["phishing_deepnn_result"] = self.clf_phishing_deepnn.classify(df)
        stats["phishing_dns_nn_result"] = self.clf_phishing_dns_nn.classify(df)
        stats["phishing_rdap_nn_result"] = self.clf_phishing_rdap_nn.classify(df)

        # Malware
        stats["malware_cnn_result"] = self.clf_malware_cnn.classify(ndf_malware).astype(float)
        # stats["malware_xgboost_result"] = self.clf_malware_xgboost.classify(ndf_malware)
        stats["malware_lgbm_result"] = self.clf_malware_lgbm.classify(df)
        stats["malware_xgboost_result"] = self.clf_malware_xgboost.classify(df)

        # DGA
        stats["dga_binary_nn_result"] = self.clf_dga_binary_nn.classify(df)
        # dga_families = self.clf_dga_multiclass_lgbm.classify(df) # not needed for training decision-maker

        # Calculate derived statistics (additional inputs for the decision making model)
        no_phishing_classifiers = 6
        no_malware_classifiers = 3
        stats["phishing_sum"] = stats["phishing_cnn_result"] + stats["phishing_lgbm_result"] + \
            stats["phishing_xgboost_result"] + stats["phishing_deepnn_result"] + stats["phishing_dns_nn_result"] + stats["phishing_rdap_nn_result"]
        stats["phishing_avg"] = stats["phishing_sum"] / no_phishing_classifiers
        stats["phishing_prod"] = stats["phishing_cnn_result"] * stats["phishing_lgbm_result"] * \
            stats["phishing_xgboost_result"] * stats["phishing_deepnn_result"] * \
            stats["phishing_dns_nn_result"] * stats["phishing_rdap_nn_result"]
        stats["malware_sum"] = stats["malware_cnn_result"] + stats["malware_lgbm_result"] + stats["malware_xgboost_result"]
        stats["malware_avg"] = stats["malware_sum"] / no_malware_classifiers
        stats["malware_prod"] = stats["malware_cnn_result"] * stats["malware_lgbm_result"] * stats["malware_xgboost_result"]
        stats["total_sum"] = stats["phishing_sum"] + stats["malware_sum"] + stats["dga_binary_nn_result"]
        stats["total_avg"] = stats["total_sum"] / (no_phishing_classifiers + no_malware_classifiers + 1)
        stats["total_prod"] = stats["phishing_prod"] * stats["malware_prod"] * stats["dga_binary_nn_result"]

        if add_final:
            # Calculate the overall badness probability
            stats["badness_probability"] = stats.apply(self.calculate_badness_probability, axis=1)

        stats["phishing_xgboost_result"] = self.clf_phishing_xgboost.classify(df)
        stats["phishing_deepnn_result"] = self.clf_phishing_deepnn.classify(df)

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
    

    def dump_ndf(self, df: pd.DataFrame, classifier_type: str, output_filename=False) -> list[dict]:
        """
        Creates an NDF representation of the input data and stores it as a file.
        """

        if classifier_type != "phishing" and classifier_type != "malware" and \
            classifier_type != "dga_binary" and classifier_type != "dga_multiclass":
            raise ValueError("Invalid classifier type") 

        # Shuffle the feature vector to the order in which it was used in training
        df = df.reindex(columns=features_in_expected_order, copy=False)

        ndf = self.pp.df_to_NDF(df, classifier_type)

        # Store as file if necessary
        if output_filename:
            joblib.dump(ndf, output_filename)

        return ndf


    def debug_domain(self, domain_name: str, df: pd.DataFrame, n_top_features: int = 10):
        """
        Debugs a single domain name by showing the most important features for decision
        """
        
        df = df.copy()

        # Drop all undesired columns
        df = df[[col for col in df.columns if col in features_in_expected_order]]

        # Rearrange the feature vector to the order in which it was used in training
        df = df.reindex(columns=features_in_expected_order, copy=False)

        ndf_phishing = self.pp.df_to_NDF(df, "phishing")

        return {
            #"phishing_cnn": self.clf_phishing_cnn.debug_domain(domain_name, ndf_phishing, n_top_features),
            "phishing_lgbm": self.clf_phishing_lgbm.debug_domain(domain_name, df, n_top_features)
            # TODO: Add explanations for other classifiers
        }


    def classify_domains(self, df: pd.DataFrame) -> list[dict]:
        """
        Classifies the domains from a pandas df and returns list the results.
        Each row of the input DF is a single domain, represented by a column  domain_name
        and 
        """
        # The domain name should be the index
        # df.set_index('domain_name', inplace=True)

        # Rearrange the feature vector to the order in which it was used in training
        df = df.reindex(columns=features_in_expected_order, copy=False)

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
        #stats["phishing_cnn_result"] = self.clf_phishing_cnn.classify(ndf_phishing).astype(float)
        stats["phishing_cnn_result"] = self.clf_phishing_cnn.classify(df)
        # stats["phishing_lgbm_result"] = self.clf_phishing_lgbm.classify(ndf_phishing)
        stats["phishing_lgbm_result"] = self.clf_phishing_lgbm.classify(df)
        stats["phishing_xgboost_result"] = self.clf_phishing_xgboost.classify(df)
        stats["phishing_deepnn_result"] = self.clf_phishing_deepnn.classify(df)
        stats["phishing_dns_nn_result"] = self.clf_phishing_dns_nn.classify(df)
        stats["phishing_rdap_nn_result"] = self.clf_phishing_rdap_nn.classify(df)

        # Malware
        stats["malware_cnn_result"] = self.clf_malware_cnn.classify(ndf_malware).astype(float)
        # stats["malware_xgboost_result"] = self.clf_malware_xgboost.classify(ndf_malware)
        stats["malware_lgbm_result"] = self.clf_malware_lgbm.classify(df)
        stats["malware_xgboost_result"] = self.clf_malware_xgboost.classify(df)

        # DGA binary
        stats["dga_binary_nn_result"] = self.clf_dga_binary_nn.classify(df)

        # Calculate derived statistics (additional inputs for the decision making model)
        no_phishing_classifiers = 6
        no_malware_classifiers = 3
        stats["phishing_sum"] = stats["phishing_cnn_result"] + stats["phishing_lgbm_result"] + \
            stats["phishing_xgboost_result"] + stats["phishing_deepnn_result"] + stats["phishing_dns_nn_result"] + stats["phishing_rdap_nn_result"]
        stats["phishing_avg"] = stats["phishing_sum"] / no_phishing_classifiers
        stats["phishing_prod"] = stats["phishing_cnn_result"] * stats["phishing_lgbm_result"] * \
            stats["phishing_xgboost_result"] * stats["phishing_deepnn_result"] * \
            stats["phishing_dns_nn_result"] * stats["phishing_rdap_nn_result"]
        stats["malware_sum"] = stats["malware_cnn_result"] + stats["malware_lgbm_result"] + stats["malware_xgboost_result"]
        stats["malware_avg"] = stats["malware_sum"] / no_malware_classifiers
        stats["malware_prod"] = stats["malware_cnn_result"] * stats["malware_lgbm_result"] * stats["malware_xgboost_result"]
        stats["total_sum"] = stats["phishing_sum"] + stats["malware_sum"] + stats["dga_binary_nn_result"]
        stats["total_avg"] = stats["total_sum"] / (no_phishing_classifiers + no_malware_classifiers + 1)
        stats["total_prod"] = stats["phishing_prod"] * stats["malware_prod"] * stats["dga_binary_nn_result"]

        # Calculate the overall badness probability
        stats["badness_probability"] = stats.apply(self.calculate_badness_probability, axis=1)

        stats["phishing_xgboost_result"] = self.clf_phishing_xgboost.classify(df)
        stats["phishing_deepnn_result"] = self.clf_phishing_deepnn.classify(df)

        # DGA Families
        stats["dga_families"] = self.clf_dga_multiclass_lgbm.classify(df)

        # print("=====================================")
        # print(stats)
        # print("=====================================")

        # Create an array of results
        results = stats.apply(lambda domain_stats: self.generate_result(domain_stats), axis=1).tolist()

        return results

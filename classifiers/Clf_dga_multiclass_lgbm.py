"""
Phishing LightGBM classifier for DomainRadar

Classifies phishing domains using the Light Gradient-Boosting Machine (LightGBM) model.
"""
__author__ = "Radek Hranicky"

import operator
import os
import pickle

from pandas import DataFrame

from classifiers.options import PipelineOptions


class Clf_dga_multiclass_lgbm:
    """
        Class for the LightGBM phishing classifier.
        Expects the model loaded in the ./models/ directory.
        Use the `classify` method to classify a dataset of domain names.
    """

    inverse_class_map = {
        0: 'dga:pushdotid', 1: 'dga:matsnu', 2: 'dga:szribi', 3: 'dga:ranbyus', 4: 'dga:suppobox',
        5: 'dga:torpig', 6: 'dga:tsifiri', 7: 'dga:pykspa2s', 8: 'dga:volatilecedar', 9: 'dga:ud2',
        10: 'dga:qadars', 11: 'dga:dyre', 12: 'dga:diamondfox', 13: 'dga:cryptolocker', 14: 'dga:madmax',
        15: 'dga:dnsbenchmark', 16: 'dga:pykspa', 17: 'dga:chir', 18: 'dga:urlzone', 19: 'dga:gameoverp2p',
        20: 'dga:ccleaner', 21: 'dga:pykspa2', 22: 'dga:gspy', 23: 'dga:makloader', 24: 'dga:bedep',
        25: 'dga:qakbot', 26: 'dga:sutra', 27: 'dga:nymaim2', 28: 'dga:pitou', 29: 'dga:dircrypt',
        30: 'dga:beebone', 31: 'dga:vawtrak', 32: 'dga:tofsee', 33: 'dga:sphinx', 34: 'dga:infy',
        35: 'dga:modpack', 36: 'dga:conficker', 37: 'dga:murofet', 38: 'dga:pandabanker', 39: 'dga:feodo',
        40: 'dga:corebot', 41: 'dga:fobber', 42: 'dga:symmi', 43: 'dga:sisron', 44: 'dga:randomloader',
        45: 'dga:wd', 46: 'dga:hesperbot', 47: 'dga:tempedreve', 48: 'dga:vidro', 49: 'dga:virut',
        50: 'dga:goznym', 51: 'dga:blackhole', 52: 'dga:nymaim', 53: 'dga:tinba', 54: 'dga:ud4',
        55: 'dga:ramnit', 56: 'dga:pushdo', 57: 'dga:qhost', 58: 'dga:chinad', 59: 'dga:bobax',
        60: 'dga:tempedrevetdd', 61: 'dga:ramdo', 62: 'dga:gameover', 63: 'dga:mirai', 64: 'dga:locky',
        65: 'dga:monerominer', 66: 'dga:simda', 67: 'dga:dnschanger', 68: 'dga:downloader', 69: 'dga:darkshell',
        70: 'dga:padcrypt', 71: 'dga:xxhex', 72: 'dga:ebury', 73: 'dga:banjori', 74: 'dga:oderoor',
        75: 'dga:dmsniff', 76: 'dga:vidrotid', 77: 'dga:xshellghost', 78: 'dga:necurs', 79: 'dga:tinynuke',
        80: 'dga:murofetweekly', 81: 'dga:qsnatch', 82: 'dga:ud3', 83: 'dga:emotet', 84: 'dga:rovnix',
        85: 'dga:proslikefan', 86: 'dga:mydoom', 87: 'dga:shifu', 88: 'dga:bamital', 89: 'dga:gozi',
        90: 'dga:redyms', 91: 'dga:omexo', 92: 'dga:ekforward'
    }

    def __init__(self, options: PipelineOptions):
        """
        Initializes the classifier.
        """

        # Load the LightGBM model
        self.model = pickle.load(open(os.path.join(options.models_dir, "dga_multiclass_lgbm_model.pkl"), "rb"))

        # Get the number of features expected by the model
        self.expected_feature_size = self.model.n_features_

    def classify(self, input_data: DataFrame) -> list:
        # Load the trained model

        # Preserve only lex_ columns
        input_data = input_data.filter(regex='^lex_')

        # Handle NaNs
        input_data.fillna(-1, inplace=True)

        # Make predictions for all domain names
        predicted_families_prob = self.model.predict_proba(input_data)

        results = []

        for prob in predicted_families_prob:
            dga_families = dict()

            # Save families with probability higher than 10%
            for family_id, family_prob in enumerate(prob):
                if family_prob > 0.1:
                    family_name = self.inverse_class_map[family_id]
                    dga_families[family_name] = family_prob

            # Sort families by probability
            dga_families = dict(sorted(dga_families.items(), key=operator.itemgetter(1), reverse=True))

            results.append(dga_families)

        return results

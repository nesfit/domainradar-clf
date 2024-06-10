"""
Phishing CNN classifier for DomainRadar

The classifier uses a convolution neural network (CNN) model to classify phishing
domain names. Feature values are transformed into a 2D matrix which is then
processed by the model and its convolution layers.
"""
__authors__ = ["Jan Polisensky (model definition & training)",
               "Radek Hranicky (supporting class, testing, integration)"]

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import shap

from .models.phishing_cnn_model_net import Net as Phishing_CNN_Net
from .options import PipelineOptions


class Clf_phishing_cnn:
    """
        Class for the CNN phishing classifier.
        Expects the model loaded in the ./models/ directory.
        Use the `classify` method to classify a dataset of domain names.
    """

    def __init__(self, options: PipelineOptions):
        """
        Initializes the classifier.
        """

        self.device = torch.device("cpu")  # Production environment uses CPU

        # The number of features in the feature vector
        # IMPORTANT: EDIT THIS if the model is changed!
        self.feature_size = 171

        # Calculate the sizes and padding of the NN model
        self.desired_size = self.next_perfect_square(self.feature_size)
        self.side_size = int(self.desired_size ** 0.5)
        self.padding = self.desired_size - self.feature_size

        # Load and evaluate the model
        self.state_dict = torch.load(os.path.join(options.models_dir, 'phishing_cnn_model_state_dict.pth'),
                                     map_location=self.device)
        self.model = Phishing_CNN_Net(self.side_size).to(self.device)
        self.model.load_state_dict(self.state_dict)
        self.model.eval()

    def next_perfect_square(self, n):
        """
        Calculates the next perfect square greater than a given number
        """
        next_square = math.ceil(n ** 0.5) ** 2
        return next_square
    

    def debug_domain(self, domain_name: str, ndf_data: dict, n_top_features: int = 10):
        """
        Debug a specific domain by calculating the feature importance for its classification.
        
        Args:
            domain_name (str): The domain name to debug.
            ndf_data (dict): The NDF data for the domain.
            n_top_features (int, optional): Number of top features to display. Default is 10.
        """
        # Ensure feature_names is a list
        feature_names = list(ndf_data['feature_names'])

        # Find the index corresponding to the domain name
        try:
            domain_index = ndf_data['domain_names'].index(domain_name)
        except ValueError:
            raise ValueError("Domain name not found in the input data.")
        
        domain_row = ndf_data['features'][domain_index]

        # Convert domain_row to tensor
        domain_row_tensor = torch.tensor(domain_row, dtype=torch.float32).unsqueeze(0)

        # Pad and reshape the tensor
        if self.padding > 0:
            domain_tensor_padded = F.pad(domain_row_tensor, (0, self.padding), 'constant', 0)
        else:
            domain_tensor_padded = domain_row_tensor

        domain_tensor_reshaped = domain_tensor_padded.view(1, 1, self.side_size, self.side_size)
        domain_tensor_reshaped = domain_tensor_reshaped.to(self.device)

        # Ensure that the background data is correctly shaped and processed
        background = torch.zeros((1, 1, self.side_size, self.side_size)).to(self.device)
        explainer = shap.DeepExplainer(self.model, background)
        shap_values = explainer.shap_values(domain_tensor_reshaped)

        # Validate the shap_values and model output
        with torch.no_grad():
            model_output = self.model(domain_tensor_reshaped).cpu().numpy()
            assert len(shap_values) == len(model_output), "SHAP values and model output length mismatch"

        # Get feature importance for the specific prediction
        domain_shap_values = shap_values[0].reshape(-1)[:len(feature_names)]  # Since DeepExplainer returns a list of arrays
        domain_feature_importance = zip(feature_names, domain_shap_values)

        # Sort features by absolute SHAP value
        sorted_feature_importance = sorted(domain_feature_importance, key=lambda x: abs(x[1]), reverse=True)

        # Get the top n features
        top_features = sorted_feature_importance[:n_top_features]

        # Store the top features and their values in a dictionary
        feature_info = []
        for feature, importance in top_features:
            feature_info.append({
                "feature": feature,
                "value": domain_row[feature_names.index(feature)],
                "shap_value": importance
            })

        # Calculate the probability for the domain
        with torch.no_grad():
            outputs = self.model(domain_tensor_reshaped)
            probabilities = F.softmax(outputs, dim=1)
            probability = probabilities[0, 1].item()  # Probability of class 1 (positive class)

        # Create data for the force plot
        force_plot_data = (explainer.expected_value[0], domain_shap_values, domain_row)

        # Return the information as a dictionary
        return {
            "top_features": feature_info,
            "probability": probability,
            "force_plot_data": force_plot_data
        }


    def classify(self, ndf_data: dict) -> np.array:
        """
        Classifies phishing domain names using the CNN model.
        The input is an NDF representation of feature vectors, one for each domain.
        Returns a numpy array of malicious class probabilities
        """

        # Get the data tensor from the NDF dataset
        data_tensor = torch.tensor(ndf_data['features'], dtype=torch.float32)

        # Verify if the shape is correct
        if data_tensor.shape[1] != self.feature_size:
            raise Exception("The number of features in the input data does not match the expected size!")

        desired_size = self.next_perfect_square(self.feature_size)

        if self.padding > 0:
            data_tensor_padded = F.pad(data_tensor, (0, self.padding), 'constant', 0)
        else:
            data_tensor_padded = data_tensor

        data_tensor_reshaped = data_tensor_padded.view(-1, 1, self.side_size, self.side_size)
        data_tensor_reshaped = data_tensor_reshaped.to(self.device)

        with torch.no_grad():
            outputs = self.model(data_tensor_reshaped)
            probabilities = F.softmax(outputs, dim=1)
            probabilities_np = probabilities.detach().cpu().numpy()
            positive_class_probabilities = probabilities_np[:, 1]  # Extract the probability of class 1 (positive class)

        return positive_class_probabilities

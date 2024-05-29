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
import torch
import torch.nn.functional as F

from .models.phishing_cnn_model_net import Net as Phishing_CNN_Net


class Clf_phishing_cnn:
    """
        Class for the CNN phishing classifier.
        Expects the model loaded in the ./models/ directory.
        Use the `classify` method to classify a dataset of domain names.
    """

    def __init__(self):
        """
        Initializes the classifier.
        """

        self.device = torch.device("cpu")  # Production environment uses CPU

        # Get the directory of the current file
        self.base_dir = os.path.dirname(__file__)

        # The number of features in the feature vector
        # IMPORTANT: EDIT THIS if the model is changed!
        self.feature_size = 171

        # Calculate the sizes and padding of the NN model
        self.desired_size = self.next_perfect_square(self.feature_size)
        self.side_size = int(self.desired_size ** 0.5)
        self.padding = self.desired_size - self.feature_size

        # Load and evaluate the model
        self.state_dict = torch.load(os.path.join(self.base_dir, 'models/phishing_cnn_model_state_dict.pth'),
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

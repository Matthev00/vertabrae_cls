import abc

import torch
from torch import nn


class VertebraeClassifier(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, num_classes: int) -> None:
        """
        VertebraeClassifier is an abstract base class for vertebrae classification models.
        Args:
            num_classes (int): Number of classes for classification.
        """
        super().__init__()
        self.num_classes = num_classes

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pass

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the class of the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted class indices.
        """
        logits = self.forward(x)
        return torch.argmax(torch.softmax(logits, dim=1), dim=1)

    def compute_loss(
        self, x: torch.Tensor, labels: torch.Tensor, criterion: nn.Module
    ) -> torch.Tensor:
        """
        Compute the loss for the given input and labels.

        Args:
            x (torch.Tensor): Input tensor.
            labels (torch.Tensor): Ground truth labels.
            criterion (nn.Module): Loss function.

        Returns:
            torch.Tensor: Computed loss.
        """
        logits = self.forward(x)
        return criterion(logits, labels)

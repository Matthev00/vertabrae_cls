import abc
from pathlib import Path

import torch
from torch import nn

from src.config import CLASS_NAMES_FILE_PATH


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

    def topk_predictions(
        self, x: torch.Tensor, class_names_path: Path, k: int = 3
    ) -> list[list[tuple[str, float]]]:
        """
        Return the top-k predicted classes and their probabilities for each input sample.

        Args:
            x (torch.Tensor): Input tensor of shape (N, ...).
            k (int): Number of top predictions to return. Defaults to 3.
            class_names_path (Path): Path to a text file containing class names (one per line).

        Returns:
            List[List[Tuple[str, float]]]: For each input sample, a list of (class_name, probability) tuples.
        """
        with open(class_names_path, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]

        if len(class_names) != self.num_classes:
            raise ValueError(
                f"Number of class names ({len(class_names)}) does not match model output ({self.num_classes})."
            )

        self.eval()
        with torch.inference_mode():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_idxs = torch.topk(probs, k=k, dim=1)

        results = []
        for sample_probs, sample_idxs in zip(top_probs, top_idxs):
            sample_result = [
                (class_names[idx], round(float(prob), 4))
                for idx, prob in zip(sample_idxs, sample_probs)
            ]
            results.append(sample_result)

        return results

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

    def load_weights(self, weights_path: Path, strict: bool = True, assign: bool = False):
        """
        Load pretrained weights into the model from a .pth file.

        Args:
            weights_path (Path): Path to the .pth file containing the saved state_dict.
            strict (bool, optional): Whether to strictly enforce that the keys in the state_dict match
                the model's keys. If False, unmatched keys will be ignored. Defaults to True.
            assign (bool, optional): If True, assigns weights without in-place modification
                (requires PyTorch >= 2.1). Defaults to False.

        Raises:
            RuntimeError: If loading fails due to mismatched keys and `strict=True`.
            TypeError: If the loaded file is not a valid state_dict.
        """
        self.load_state_dict(torch.load(weights_path), strict, assign)

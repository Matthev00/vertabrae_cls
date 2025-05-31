import torch
from monai.bundle import ConfigParser
from torch import nn

from src.config import SEG_MODEL_DIR
from src.modeling.base_model import VertebraeClassifier


class SegResNetClassifier(VertebraeClassifier):
    def __init__(self, num_classes: int = 10) -> None:
        """
        Classifier using MONAI's SegResNet backbone.

        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__(num_classes)
        self._load_monai_model()
        self._delete_decoder_layers()
        self._add_classifier_layers()

    def _load_monai_model(self) -> None:
        """
        Load the MONAI SegResNet model from the specified directory.
        """
        config_path = SEG_MODEL_DIR / "configs/inference.json"
        parser = ConfigParser()
        parser.read_config(config_path)

        self.model = parser.get_parsed_content("network_def")
        weights = torch.load(SEG_MODEL_DIR / "models/model.pt", map_location="cpu")
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        self.model.load_state_dict(weights)

    def _delete_decoder_layers(self) -> None:
        """
        Remove decoder layers from the model to keep only the encoder part.
        """
        for attr in ["up_samples", "up_conv", "final_conv"]:
            if hasattr(self.model, attr):
                delattr(self.model, attr)

    def _add_classifier_layers(self) -> None:
        """
        Add classifier layers to the model.
        """
        self.model.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.model.flatten = nn.Flatten()
        self.model.classifier = nn.Linear(256, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.model.convInit(x)
        for down in self.model.down_layers:
            x = down(x)
        x = self.model.pool(x)
        x = self.model.flatten(x)
        x = self.model.classifier(x)
        return x

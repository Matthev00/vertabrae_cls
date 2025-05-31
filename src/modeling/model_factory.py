from typing import Literal

import torch

from src.modeling.base_model import VertebraeClassifier
from src.modeling.med3_transfer import Med3DClassifier
from src.modeling.monai_resnet import SegResNetClassifier


def create_model(
    model_type: Literal["med3d", "monai", "base"],
    num_classes: int,
    device: torch.device = torch.device("cuda"),
    **kwargs: dict,
) -> VertebraeClassifier:
    """
    Factory function that returns a ready-to-use vertebrae classification model.

    Args:
        model_type (str): Type of model to create ("med3d", "monai", "resnet").
        num_classes (int): Number of output classes.
        device (torch.device): Target device (default: "cuda").
        **kwargs (dict): Additional parameters for model configuration.

    Returns:
        VertebraeClassifier: Ready model in eval mode on given device.
    """
    if model_type == "med3d":
        model = Med3DClassifier(
            num_classes=num_classes,
            device=device,
            load_pretrained=True,
            **kwargs,
        )
    elif model_type == "monai":
        model = SegResNetClassifier(num_classes=num_classes)
    elif model_type == "base":
        model = Med3DClassifier(
            num_classes=num_classes,
            device=device,
            load_pretrained=False,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    model.eval()
    return model

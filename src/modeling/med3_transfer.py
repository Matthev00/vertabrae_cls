from typing import Literal

import torch
from huggingface_hub import hf_hub_download

from src.config import MODELS_DIR
from src.modeling.base_model import VertebraeClassifier
from src.modeling.resnet3d import get_resnet


class Med3DClassifier(VertebraeClassifier):
    def __init__(
        self,
        num_classes: int,
        model_depth: int = 18,
        shortcut_type: Literal["A", "B"] = "B",
        load_pretrained: bool = True,
        freeze_backbone: bool = True,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """
        Classifier using Med3D ResNet backbone with optional pretrained weights.

        Args:
            num_classes (int): Number of output classes.
            model_depth (int): Depth of ResNet (e.g., 10, 18, 34...).
            shortcut_type (Literal['A', 'B']): Type of shortcut connection.
            load_pretrained (bool): Whether to load pretrained weights from Med3D.
            freeze_backbone (bool): Whether to freeze the backbone weights.
        """
        super().__init__(num_classes)
        self.device = device

        self.model = get_resnet(
            model_depth=model_depth,
            num_classes=num_classes,
            shortcut_type=shortcut_type,
        )

        if load_pretrained:
            self._load_med3d_weights(model_depth)

        if freeze_backbone:
            self._freeze_backbone_weights()

    def _load_med3d_weights(self, model_depth: int) -> None:
        """
        Load pretrained Med3D weights from HuggingFace, skipping the classification layer.
        """
        hf_mapping = {
            10: ("TencentMedicalNet/MedicalNet-ResNet10", "resnet_10.pth"),
            18: ("TencentMedicalNet/MedicalNet-ResNet18", "resnet_18.pth"),
            34: ("TencentMedicalNet/MedicalNet-ResNet34", "resnet_34.pth"),
            50: ("TencentMedicalNet/MedicalNet-ResNet50", "resnet_50.pth"),
            101: ("TencentMedicalNet/MedicalNet-ResNet101", "resnet_101.pth"),
            152: ("TencentMedicalNet/MedicalNet-ResNet152", "resnet_152.pth"),
            200: ("TencentMedicalNet/MedicalNet-ResNet200", "resnet_200.pth"),
        }

        if model_depth not in hf_mapping:
            raise ValueError(f"No pretrained weights available for model depth {model_depth}")

        repo_id, filename = hf_mapping[model_depth]
        weight_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=MODELS_DIR / "med3d"
        )

        state_dict = torch.load(weight_path, map_location=self.device)

        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
            if not k.startswith("conv-seg")
        }

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(
            f"[Med3D] Loaded weights with {len(missing)} missing and {len(unexpected)} unexpected keys."
        )

    def _freeze_backbone_weights(self) -> None:
        """
        Freeze the backbone weights to prevent them from being updated during training.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

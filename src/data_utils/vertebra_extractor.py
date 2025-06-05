import os
import tempfile
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch
from monai.bundle import ConfigParser
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
)

from src.config import SEG_MODEL_DIR, VERTEBRAE_MAP


class VertebraExtractor:
    """
    A class to extract specific vertebrae from CT images.
    """

    def __init__(
        self, is_full_resolution: bool = False, device: torch.device = torch.device("cuda")
    ):
        """
        Initialize the VertebraExtractor.

        Args:
            is_full_resolution (bool): If True, use full resolution model.
            device (torch.device): The device to run the model on.
        """
        pixdim = (1.5, 1.5, 1.5) if is_full_resolution else (3.0, 3.0, 3.0)
        self.transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureTyped(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),
                NormalizeIntensityd(keys=["image"], nonzero=True),
                ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
            ]
        )
        self.device = device
        self.is_full_resolution = is_full_resolution
        self.model = self._load_model()

        self.segmentation: torch.Tensor | None = None
        self.transformed_input_tensor: torch.Tensor | None = None

    def _load_model(self) -> torch.nn.Module:
        """
        Load the vertebra extraction model.

        Returns:
            torch.nn.Module: The loaded model.
        """

        model_filename = "model.pt" if self.is_full_resolution else "model_lowres.pt"
        model_path = SEG_MODEL_DIR / "models" / model_filename
        config_path = SEG_MODEL_DIR / "configs" / "inference.json"

        parser = ConfigParser()
        parser.read_config(config_path)

        model = parser.get_parsed_content("network_def")
        weights = torch.load(model_path, map_location=self.device)

        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        model.load_state_dict(weights)
        model.eval()
        model.to(self.device)
        return model

    def _save_tensor_as_nrrd(self, tensor: torch.Tensor, metadata: dict, filename: str) -> None:
        """
        Save a PyTorch tensor as an NRRD file with metadata.

        Args:
            tensor (torch.Tensor): The tensor to save.
            metadata (dict): Metadata to include in the NRRD file.\
            filename (str): The filename to save the NRRD file as.
        """
        array = tensor.numpy()

        image = sitk.GetImageFromArray(array)

        if "spacing" in metadata:
            image.SetSpacing(metadata["spacing"])
        if "direction" in metadata:
            image.SetDirection(metadata["direction"])
        if "origin" in metadata:
            image.SetOrigin(metadata["origin"])

        sitk.WriteImage(image, filename)

    def _transform_data(self, tensor: torch.Tensor, metadata: dict) -> torch.Tensor:
        """
        Transform the input tensor data.

        Args:
            tensor (torch.Tensor): The input tensor.
            metadata (dict): Metadata for the tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "output.nrrd")

            self._save_tensor_as_nrrd(tensor, metadata, temp_file)

            data = {"image": temp_file}

            transformed_data = self.transforms(data)
            return transformed_data["image"]

    def get_segmentation(
        self, raw_tensor: torch.Tensor, metadata: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the segmentation of the input tensor.

        Args:
            raw_tensor (torch.Tensor): The input tensor.
            metadata (dict): Metadata for the tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The segmentation tensor and the input tensor.
        """
        transformed_tensor = self._transform_data(raw_tensor, metadata)

        with torch.inference_mode():
            input_tensor = transformed_tensor.unsqueeze(0).to(self.device)
            output = sliding_window_inference(
                inputs=input_tensor,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=self.model,
                overlap=0.5,
                mode="gaussian",
            )
            output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        return output, input_tensor

    def _resize_vertebrae_tensor(
        self,
        input_tensor: torch.Tensor,
        coords: np.ndarray,
        target_size: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Resize the vertebrae tensor to the target size, applying padding if necessary.

        Args:
            input_tensor (torch.Tensor): The input tensor.
            coords (np.ndarray): Coordinates of the vertebrae.
            target_size (tuple[int, int, int]): Target size for resizing.

        Returns:
            torch.Tensor: The resized tensor with padding if required.
        """
        target_coords = np.array((coords.max(axis=1) + coords.min(axis=1)) // 2)
        z_center, y_center, x_center = target_coords

        z_half, y_half, x_half = [size // 2 for size in target_size]

        z_start = z_center - z_half
        y_start = y_center - y_half
        x_start = x_center - x_half

        z_end = z_start + target_size[0]
        y_end = y_start + target_size[1]
        x_end = x_start + target_size[2]

        pad_z_start = max(0, -z_start)
        pad_y_start = max(0, -y_start)
        pad_x_start = max(0, -x_start)

        pad_z_end = max(0, z_end - input_tensor.shape[2])
        pad_y_end = max(0, y_end - input_tensor.shape[3])
        pad_x_end = max(0, x_end - input_tensor.shape[4])

        z_start = max(0, z_start)
        y_start = max(0, y_start)
        x_start = max(0, x_start)

        z_end = min(input_tensor.shape[2], z_end)
        y_end = min(input_tensor.shape[3], y_end)
        x_end = min(input_tensor.shape[4], x_end)

        cropped_tensor = input_tensor[:, :, z_start:z_end, y_start:y_end, x_start:x_end]

        if any([pad_z_start, pad_y_start, pad_x_start, pad_z_end, pad_y_end, pad_x_end]):
            cropped_tensor = torch.nn.functional.pad(
                cropped_tensor,
                (pad_x_start, pad_x_end, pad_y_start, pad_y_end, pad_z_start, pad_z_end),
                mode="constant",
                value=0,
            )

        return cropped_tensor.detach().cpu()

    def extract_vertebrae_with_neighbors(
        self,
        input_tensor: torch.Tensor,
        metadata: dict,
        target_vertebrae: str,
        target_size: Optional[tuple[int, int, int]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Extract the specified vertebrae and its neighbors from the input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor.
            metadata (dict): Metadata for the tensor.
            target_vertebrae (str): The vertebrae to extract.
            target_size (tuple[int, int, int], optional): The target size for resizing.
                If None, the original size will be used.

        Returns:
            Optional[torch.Tensor]: The extracted vertebrae tensor, or None if no vertebrae are found.
        """
        target_label = VERTEBRAE_MAP.get(target_vertebrae)
        if target_label is None:
            raise ValueError(f"Invalid target vertebrae: {target_vertebrae}")

        labels = [target_label]
        if target_size is None:
            labels.extend([max(18, target_label - 1), min(41, target_label + 1)])

        if self.segmentation is None or self.transformed_input_tensor is None:
            self.segmentation, self.transformed_input_tensor = self.get_segmentation(
                input_tensor, metadata
            )

        mask = np.isin(self.segmentation, labels).astype(np.uint8)

        coords = np.array(mask.nonzero())
        if coords.shape[1] == 0:
            print("No vertebrae found in the segmentation.")
            return None

        zmin, ymin, xmin = coords.min(axis=1)
        zmax, ymax, xmax = coords.max(axis=1)

        if target_size is None:
            output = self.transformed_input_tensor[:, :, zmin:zmax, ymin:ymax, xmin:xmax]
            return output.squeeze().detach().cpu()

        return self._resize_vertebrae_tensor(self.transformed_input_tensor, coords, target_size)

import os
from pathlib import Path
from typing import Optional

import SimpleITK as sitk
import torch


class DICOMReader:
    """
    Class to read DICOM files and convert them to PyTorch tensors.
    """

    def __init__(self, patient_dir: Path):
        """
        Initialize the DICOMReader with the directory containing DICOM files.

        Args:
            patient_dir (Path): Directory containing DICOM files.
        """
        self.patient_dir = patient_dir

    def _get_image_metadata(self, image: sitk.Image) -> dict:
        """
        Extract metadata from a SimpleITK image.

        Args:
            image (sitk.Image): The SimpleITK image object.

        Returns:
            dict: A dictionary containing the image metadata.
        """
        metadata = {
            "spacing": image.GetSpacing(),
            "direction": image.GetDirection(),
            "origin": image.GetOrigin(),
        }
        return metadata

    def _convert_dicom_to_tensor(self, image: sitk.Image) -> torch.Tensor:
        """
        Convert a SimpleITK image to a PyTorch tensor.

        Args:
            image (sitk.Image): The SimpleITK image object.

        Returns:
            torch.Tensor: The converted PyTorch tensor.
        """
        image_array = sitk.GetArrayFromImage(image)
        tensor = torch.tensor(image_array, dtype=torch.float32)
        return tensor

    def _get_series_description(self, first_file: str) -> str:
        """
        Extract the series description from the first DICOM file.

        Args:
            first_file (str): The path to the first DICOM file.

        Returns:
            str: The series description.
        """
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(first_file)
        file_reader.ReadImageInformation()
        return (
            file_reader.GetMetaData("0008|103e")
            if file_reader.HasMetaDataKey("0008|103e")
            else "Unknown"
        )

    def load_dicom_series_with_metadata(self, series_dir: Path) -> tuple[torch.Tensor, str, dict]:
        """
        Read a DICOM series from a directory and return the image tensor, series description, and metadata.

        Args:
            series_dir (str): Path to the directory containing DICOM series.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The image tensor.
                - str: The series description.
                - dict: Metadata including spacing, direction, and origin.

        Raises:
            ValueError: If no DICOM series is found in the directory.
        """
        reader = sitk.ImageSeriesReader()
        dicom_series = reader.GetGDCMSeriesFileNames(series_dir)
        if not dicom_series:
            raise ValueError(f"Did not find any DICOM Series in: {series_dir}")
        reader.SetFileNames(dicom_series)

        image = reader.Execute()
        tensor = self._convert_dicom_to_tensor(image)

        metadata = self._get_image_metadata(image)

        series_description = self._get_series_description(dicom_series[0])

        return tensor, series_description, metadata

    def _load_all_dicom_series(self) -> dict[str, dict[str, torch.Tensor]]:
        """
        Loads all DICOM series from a patient folder and organizes them by series description.

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: A dictionary where keys are series descriptions,
            and values are dictionaries containing tensors and metadata.
        """
        series_tensors = {}

        for root, dirs, files in os.walk(self.patient_dir):
            if files:
                try:
                    tensor, series_description, metadata = self.load_dicom_series_with_metadata(
                        root
                    )
                    series_tensors[series_description] = {"tensor": tensor, "metadata": metadata}
                except Exception:
                    pass

        return series_tensors

    def _select_tensor_by_priority(
        self, series_tensors: dict[str, dict[str, torch.Tensor]]
    ) -> tuple[Optional[torch.Tensor], Optional[str], Optional[dict]]:
        """
        Selects a tensor based on priority: "kosci" > "miekkie" > largest tensor.

        Args:
            series_tensors (Dict[str, Dict[str, torch.Tensor]]): Dictionary of series descriptions
            with tensors and metadata.

        Returns:
            Tuple[Optional[torch.Tensor], Optional[str], Optional[Dict]]: The selected tensor,
            its description, and metadata. Returns None if no tensor is found.
        """
        selected_tensor = None
        selected_description = None
        selected_metadata = None

        for series_description, data in series_tensors.items():
            if "kosci" in series_description.lower():
                return data["tensor"], series_description, data["metadata"]

        for series_description, data in series_tensors.items():
            if "miekkie" in series_description.lower():
                return data["tensor"], series_description, data["metadata"]

        max_size = 0
        for series_description, data in series_tensors.items():
            tensor_size = data["tensor"].numel()
            if tensor_size > max_size:
                max_size = tensor_size
                selected_tensor = data["tensor"]
                selected_metadata = data["metadata"]
                selected_description = series_description

        return selected_tensor, selected_description, selected_metadata

    def process_dicom_series(self) -> tuple[Optional[torch.Tensor], Optional[str], Optional[dict]]:
        """
        Processes DICOM series from a root folder, selects a tensor based on priority.

        Args:
            dicom_root_folder (str): Path to the root folder containing DICOM series.

        Returns:
            Tuple[Optional[torch.Tensor], Optional[str], Optional[dict]]: The selected tensor,
            its description, and metadata. Returns None if no tensor is found.

        Raises:
            ValueError: If no DICOM series are found in the directory.
        """
        series_tensors = self._load_all_dicom_series()

        return self._select_tensor_by_priority(series_tensors)


if __name__ == "__main__":

    dicom_root_folder = "/media/mateusz/DATA/downloads/inz/MM M 62"
    dicom_reader = DICOMReader(dicom_root_folder)
    tensor, description, metadata = dicom_reader.process_dicom_series()
    print(f"Description: {description}")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Metadata: {metadata}")

import ast
import csv
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from src.config import (
    DEVICE,
    DICOM_DATA_DIR,
    IS_FULL_RESOLUTION,
    LABELS_FILE_PATH,
    TENSOR_DIR,
)
from src.data_utils.dicom_reader import DICOMReader
from src.data_utils.vertebra_extractor import VertebraExtractor


class DatasetCreator:
    """
    Class to create datasets for vertebrae segmentation.
    """

    def __init__(self, raport_file_path: Path = None) -> None:
        """
        Initialize the DatasetCreator.

        Args:
            raport_file_path (Path): Path to raport file.
        """
        self.raport_file_path = raport_file_path

    def process_patient(
        self,
        dir_name: str,
        injuried_vertebrae: list[tuple[str, str]],
        target_size: Optional[tuple[int, int, int]] = None,
    ) -> list[dict]:
        """
        Process a patient's DICOM files and extract vertebrae data.
        This function reads the DICOM files, extracts the vertebrae and their injuries,
        and returns a list of dictionaries containing the vertebrae data.
        The dictionaries contain the following keys:
            - "vertebra": The name of the vertebra.
            - "injury_type": The type of injury.
            - "target_tensor": The extracted tensor for the vertebra.
            - "II": The patient ID.

        Args:
            dir_name (str): Directory name of the patient.
            injuried_vertebrae (list[tuple[str, str]]): List of tuples containing vertebrae information (vertebra, injury_type).

        Returns:
            list[dict]: List of dictionaries containing vertebrae data.
        """
        patient_data = []
        dir_path = DICOM_DATA_DIR / dir_name
        dicom_reader = DICOMReader(dir_path)
        tensor, description, metadata = dicom_reader.process_dicom_series()
        vertebra_extractor = VertebraExtractor(IS_FULL_RESOLUTION, DEVICE)
        for vertebra, injury_type in injuried_vertebrae:
            target_tensor = vertebra_extractor.extract_vertebrae_with_neighbors(
                input_tensor=tensor,
                metadata=metadata,
                target_vertebrae=vertebra,
                target_size=target_size,
            )
            if target_tensor is not None:
                patient_data.append(
                    {
                        "vertebra": vertebra,
                        "injury_type": injury_type,
                        "target_tensor": target_tensor,
                        "II": dir_name.split(" ")[0],
                    }
                )
        return patient_data

    def extract_dir_names(raw: str) -> list[str]:
        """
        Extracts dir name from raw name

        Args:
            raw (str): raw name.

        Returns
            list[dict]: List of names extracted from raw name.
        """
        parts = raw.split()
        if "," in raw:
            prefix = " ".join(parts[:2])
            numbers = parts[2].split(",")
            if len(parts) == 4:
                numbers += [parts[3]]
            else:
                numbers[-1] = numbers[-1].strip(",")
                numbers = [n.strip() for n in numbers]
                numbers[-1] += " " + parts[-1]
            return [f"{prefix} {n.strip()} {parts[-1]}" for n in numbers[:-1]]
        return [raw]

    def get_max_index(self) -> int:
        """
        Gets max existing tensor index.

        Returns:
            int: max existing tensor index.
        """
        existing_files = list(TENSOR_DIR.glob("*.pt"))
        if existing_files:
            max_index = max(int(f.stem) for f in existing_files if f.stem.isdigit())
            return max_index
        return 0

    def save_patient_data(self, patient_data: list[dict]) -> None:
        """
        Save processed vertebra data as tensors and append metadata to a CSV file.

        Each tensor is saved to `TENSOR_DIR` using a sequential 5-digit filename (e.g., "00001.pt").
        Corresponding metadata including vertebra name, injury type, patient ID (II), and file path
        is appended to the CSV file at `LABELS_FILE_PATH`. If the CSV does not exist, it is created.

        This function is designed to support resumable dataset creation — partial results are not lost
        in case of interruption.

        Args:
            patient_data (list[dict]):
                A list of dictionaries, where each dictionary contains:
                - "target_tensor" (torch.Tensor): The extracted vertebra tensor.
                - "vertebra" (str): Name of the vertebra (e.g., "L1").
                - "injury_type" (str): Injury classification (e.g., "A1").
                - "II" (str): Patient ID prefix.
        """
        labels_file_exists = LABELS_FILE_PATH.exists()
        next_index = self.get_max_index()

        with open(LABELS_FILE_PATH, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=["vertebra", "injury_type", "II", "tensor_path"]
            )

            if not labels_file_exists:
                writer.writeheader()

            for entry in patient_data:
                tensor = entry["target_tensor"]
                vertebra = entry["vertebra"]
                injury_type = entry["injury_type"]
                II = entry["II"]

                tensor_file_path = TENSOR_DIR / f"{next_index:05d}.pt"

                TENSOR_DIR.mkdir(parents=True, exist_ok=True)

                torch.save(tensor, tensor_file_path)

                writer.writerow(
                    {
                        "vertebra": vertebra,
                        "injury_type": injury_type,
                        "II": II,
                        "tensor_path": str(tensor_file_path),
                    }
                )
                next_index += 1

    def create_dataset(self, target_size: Optional[tuple[int, int, int]] = None) -> None:
        """
        Parse the report file and process DICOM data for each patient.

        This function reads a CSV file where each row contains:
            - a patient directory identifier (may represent multiple patients),
            - a list of injured vertebrae with injury types.

        It extracts vertebra data for each patient and saves it using `process_patient`
        and `save_patient_data`. If any part of the process fails for a patient,
        others will still be processed.

        Args:
            target_size (Optional[tuple[int, int, int]]):
                Optional shape to which extracted vertebra tensors should be resized.
                If None, original size is preserved.
        """
        with open(self.raport_file_path, newline="") as f:
            reader = csv.reader(f)
            for row in tqdm(reader):
                raw_name, raw_injuries = row
                raw_name = raw_name.strip('"')
                dir_names = self.extract_dir_names(raw_name)
                injuries = ast.literal_eval(raw_injuries)
                for name in dir_names:
                    patient_data = self.process_patient(
                        dir_name=name, injuried_vertebrae=injuries, target_size=target_size
                    )
                    self.save_patient_data(patient_data)


if __name__ == "__main__":
    x = DatasetCreator()
    list = x.process_patient(dir_name="MM M 62", injuried_vertebrae=[("L5", "A0")])
    for dict in list:
        print(dict["II"])
        print(dict["vertebra"])
        print(dict["injury_type"])
        print(dict["target_tensor"].shape)

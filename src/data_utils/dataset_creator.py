import ast
import csv
import random
import shutil
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
    VERTEBRAE_MAP,
)
from src.data_utils.dicom_reader import DICOMReader
from src.data_utils.vertebra_extractor import VertebraExtractor
from src.utils import get_logger

logger = get_logger(__name__, Path("logs/dataset.txt"))


class DatasetCreator:
    """
    Class to create datasets for vertebrae segmentation.
    """

    def __init__(self, raport_file_path: Path) -> None:
        """
        Initialize the DatasetCreator.

        Args:
            raport_file_path (Path): Path to raport file.
        """
        self.raport_file_path = raport_file_path

    def _process_injured_vertebrae(
        self,
        vertebra_extractor: VertebraExtractor,
        tensor: torch.Tensor,
        metadata: dict,
        dir_name: str,
        injuried_vertebrae: list[tuple[str, str]],
        target_size: Optional[tuple[int, int, int]],
    ) -> list[dict]:
        """
        Process injured vertebrae and extract their tensors.

        Args:
            vertebra_extractor (VertebraExtractor): The vertebra extractor instance.
            tensor (torch.Tensor): The input tensor.
            metadata (dict): Metadata for the tensor.
            dir_name (str): Directory name of the patient.
            injuried_vertebrae (list[tuple[str, str]]): List of tuples containing vertebrae information (vertebra, injury_type).
            target_size (tuple[int, int, int], optional): Target size for the extracted tensors.

        Returns:
            list[dict]: List of dictionaries containing injured vertebrae data.
        """
        injured_data = []
        for vertebra, injury_type in injuried_vertebrae:
            try:
                target_tensor = vertebra_extractor.extract_vertebrae_with_neighbors(
                    input_tensor=tensor,
                    metadata=metadata,
                    target_vertebrae=vertebra,
                    target_size=target_size,
                )
                if target_tensor is not None:
                    injured_data.append(
                        {
                            "vertebra": vertebra,
                            "injury_type": injury_type,
                            "target_tensor": target_tensor,
                            "II": dir_name,
                        }
                    )
                else :
                    logger.warning(
                        f"Target tensor for {vertebra} in {dir_name} is None. Skipping."
                    )
            except Exception as e:
                logger.error(f"Error ijuried processing {vertebra} in {dir_name}: {e}")

        return injured_data

    def _process_healthy_vertebrae(
        self,
        vertebra_extractor: VertebraExtractor,
        tensor: torch.Tensor,
        metadata: dict,
        dir_name: str,
        injuried_vertebrae: list[tuple[str, str]],
        target_size: Optional[tuple[int, int, int]],
        num_healthy: int,
    ) -> list[dict]:
        """
        Process healthy vertebrae and extract their tensors.

        Args:
            vertebra_extractor (VertebraExtractor): The vertebra extractor instance.
            tensor (torch.Tensor): The input tensor.
            metadata (dict): Metadata for the tensor.
            dir_name (str): Directory name of the patient.
            injuried_vertebrae (list[tuple[str, str]]): List of tuples containing vertebrae information (vertebra, injury_type).
            target_size (tuple[int, int, int], optional): Target size for the extracted tensors.
            num_healthy (int): Number of healthy vertebrae to extract.

        Returns:
            list[dict]: List of dictionaries containing healthy vertebrae data.
        """
        healthy_data = []
        all_vertebrae = set(VERTEBRAE_MAP.keys())
        injured_vertebrae_names = {vertebra for vertebra, _ in injuried_vertebrae}
        healthy_vertebrae = list(all_vertebrae - injured_vertebrae_names)

        random.shuffle(healthy_vertebrae)

        for vertebra in healthy_vertebrae:
            if len(healthy_data) >= num_healthy:
                break

            try:
                target_tensor = vertebra_extractor.extract_vertebrae_with_neighbors(
                    input_tensor=tensor,
                    metadata=metadata,
                    target_vertebrae=vertebra,
                    target_size=target_size,
                )
                if target_tensor is not None:
                    healthy_data.append(
                        {
                            "vertebra": vertebra,
                            "injury_type": "H",
                            "target_tensor": target_tensor,
                            "II": dir_name,
                        }
                    )
            except Exception as e:
                logger.error(f"Error processing healthy {vertebra} in {dir_name}: {e}")

        if len(healthy_data) < num_healthy:
            logger.warning(
                f"Only {len(healthy_data)} healthy vertebrae extracted out of {num_healthy} requested for patient {dir_name}."
            )

        return healthy_data

    def process_patient(
        self,
        dir_name: str,
        injuried_vertebrae: list[tuple[str, str]],
        target_size: Optional[tuple[int, int, int]] = None,
        num_healthy: int = 1,
    ) -> list[dict]:
        """
        Process a patient's DICOM files and extract vertebrae data.
        This function reads the DICOM files, extracts the vertebrae and their injuries,
        and returns a list of dictionaries containing the vertebrae data.
        Additionally, it extracts a specified number of random healthy vertebrae not listed as injured.

        Args:
            dir_name (str): Directory name of the patient.
            injuried_vertebrae (list[tuple[str, str]]): List of tuples containing vertebrae information (vertebra, injury_type).
            target_size (tuple[int, int, int], optional): Target size for the extracted tensors.
            num_healthy (int): Number of healthy vertebrae to extract.

        Returns:
            list[dict]: List of dictionaries containing vertebrae data.
        """
        patient_data = []
        dir_path = DICOM_DATA_DIR / dir_name
        dicom_reader = DICOMReader(dir_path)
        tensor, description, metadata = dicom_reader.process_dicom_series()
        vertebra_extractor = VertebraExtractor(IS_FULL_RESOLUTION, DEVICE)

        patient_data.extend(
            self._process_injured_vertebrae(
                vertebra_extractor, tensor, metadata, dir_name, injuried_vertebrae, target_size
            )
        )

        patient_data.extend(
            self._process_healthy_vertebrae(
                vertebra_extractor,
                tensor,
                metadata,
                dir_name,
                injuried_vertebrae,
                target_size,
                num_healthy,
            )
        )

        return patient_data

    def extract_dir_name(self, dir_id: str) -> str:
        """
        Finds a single directory in DICOM_DATA_DIR whose name contains the given dir_id
        as the second space-separated component.

        Args:
            dir_id (str): The directory ID to match (e.g. '303').

        Returns:
            str: The matching directory name.

        Raises:
            FileNotFoundError: If no matching directory is found.
        """
        for path in DICOM_DATA_DIR.iterdir():
            if not path.is_dir():
                continue
            parts = path.name.split()
            if len(parts) >= 2 and parts[1] == dir_id:
                return str(path.name)
        raise FileNotFoundError(
            f"No directory found in {DICOM_DATA_DIR} with ID '{dir_id}' as second part."
        )

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
                - "II" (str): Patient ID.
        """
        labels_file_exists = LABELS_FILE_PATH.exists()
        next_index = self.get_max_index() + 1

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
                        "tensor_path": str(f"{next_index:05d}"),
                    }
                )
                next_index += 1

    def _get_processed_patients(self) -> set[str]:
        """
        Get processed patients from the labels file.

        Returns:
            set[str]: Set of processed patient identifiers.
        """
        processed_patients = set()
        if LABELS_FILE_PATH.exists():
            with open(LABELS_FILE_PATH, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    processed_patients.add(row["II"])
        return processed_patients

    def create_dataset(
        self,
        target_size: Optional[tuple[int, int, int]] = None,
        num_healthy: int = 1,
        ignore_existing: bool = False,
    ) -> None:
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
            num_healthy (int): Number of healthy vertebrae to extract.
            ignore_existing (bool): If True, existing extracted data will be deleted.

        """
        if ignore_existing:
            if LABELS_FILE_PATH.exists():
                LABELS_FILE_PATH.unlink()
            if TENSOR_DIR.exists():
                shutil.rmtree(TENSOR_DIR)
                print(f"Deleted contents of tensor directory: {TENSOR_DIR}")

        processed_patients = self._get_processed_patients()

        with open(self.raport_file_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in tqdm(reader):
                _, raw_injuries, dir_id, *extra = row
                logger.info(f"Processing dir id:{dir_id}.")
                try:
                    dir_name = self.extract_dir_name(dir_id)
                except FileNotFoundError as e:
                    logger.error(e)
                    continue
                if dir_name in processed_patients:
                    continue
                injuries = ast.literal_eval(raw_injuries)

                patient_data = self.process_patient(
                    dir_name=dir_name,
                    injuried_vertebrae=injuries,
                    target_size=target_size,
                    num_healthy=num_healthy,
                )
                self.save_patient_data(patient_data)


def create_dataset():
    from src.config import RAPORT_FILE_PATH, TARGET_TENSOR_SIZE

    x = DatasetCreator(RAPORT_FILE_PATH)
    x.create_dataset(target_size=TARGET_TENSOR_SIZE, num_healthy=2)

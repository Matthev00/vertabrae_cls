from pathlib import Path
from typing import Literal

import torch

from src.config import (
    CLASS_NAMES_FILE_PATH_BINARY,
    DEVICE,
    DICOM_DATA_DIR,
    IS_FULL_RESOLUTION,
    TARGET_TENSOR_SIZE,
    VERTEBRAE_MAP,
)
from src.data_utils.dicom_reader import DICOMReader
from src.data_utils.vertebra_extractor import VertebraExtractor
from src.modeling.model_factory import create_model


class InjuryDetector:
    """
    A high-level wrapper for vertebrae injury classification from DICOM series.

    This class handles the full pipeline from loading a DICOM series, extracting vertebrae
    fragments, batching them, running classification, and returning the top-k predictions
    along with any vertebrae that could not be extracted.
    """

    def __init__(
        self,
        cls_type: Literal["med3d", "monai", "base"],
        cls_path: Path,
        num_classes: int = 10,
        **model_kwargs: dict,
    ):
        """
        Initialize the injury detector with a specified classifier and weights.

        Args:
            cls_type (str): Type of classifier to use ("med3d", "monai", "base").
            cls_path (Path): Path to the pretrained weights (.pth file).
            num_classes (int): Number of output classes for classification. Defaults to 9.
            **model_kwargs (dict): Additional keyword arguments passed to the model factory.
        """
        self.classifier = create_model(
            model_type=cls_type, num_classes=num_classes, device=DEVICE, **model_kwargs
        )
        self.classifier.load_weights(weights_path=cls_path)

    def predict(
        self,
        patient_dicom_path: Path,
        k: int = 2,
    ) -> tuple[list[dict], set[str]]:
        """
        Predict the top-k class labels for each vertebra in a patient's DICOM series.

        This method reads a DICOM directory, extracts individual vertebrae (with neighbors),
        batches them, performs classification using the loaded model, and returns predictions
        and missing vertebrae.

        Args:
            patient_dicom_path (Path): Path to the patient's DICOM folder (relative to DICOM_DATA_DIR).
            k (int): Number of top class predictions to return per vertebra. Defaults to 2.

        Returns:
            Tuple[list[dict], set[str]]:
                - A list of dictionaries for each detected vertebra, each with:
                    {
                        "vertebra": <vertebra_label>,
                        "topk": [ (class_name, probability), ... ]
                    }
                - A set of vertebra labels that could not be extracted from the DICOM data.
        """
        vertebra_extractor = VertebraExtractor(IS_FULL_RESOLUTION, DEVICE)
        dicom_reader = DICOMReader(DICOM_DATA_DIR / patient_dicom_path)
        tensor, description, metadata = dicom_reader.process_dicom_series()

        all_vertebrae = sorted(VERTEBRAE_MAP.keys())
        vertebra_tensors = []
        vertebra_names = []
        unfound_vertebrae = set()

        for vertebra in all_vertebrae:
            vt = vertebra_extractor.extract_vertebrae_with_neighbors(
                input_tensor=tensor,
                metadata=metadata,
                target_vertebrae=vertebra,
                target_size=TARGET_TENSOR_SIZE,
            )
            if vt is None:
                unfound_vertebrae.add(vertebra)
                continue
            if vt.dim() == 5 and vt.shape[0] == 1:
                vt = vt.squeeze(0)
            vertebra_tensors.append(vt)
            vertebra_names.append(vertebra)

        if not vertebra_tensors:
            return [], unfound_vertebrae

        batch = torch.stack(vertebra_tensors).to(DEVICE)
        topk_results = self.classifier.topk_predictions(batch, CLASS_NAMES_FILE_PATH_BINARY, k)

        patient_predictions = [
            {"vertebra": vertebra, "topk": topk}
            for vertebra, topk in zip(vertebra_names, topk_results)
        ]

        return patient_predictions, unfound_vertebrae

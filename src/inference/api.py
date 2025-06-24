from pathlib import Path
from typing import List, Tuple

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import DICOM_DATA_DIR, CLS_MODEL_DIR
from src.inference.predictor import InjuryDetector

app = FastAPI(title="Vertebrae Injury Classification API")

detector = InjuryDetector(
    cls_type="med3d",
    cls_path=CLS_MODEL_DIR / "polar-oath-15_best.pt",
    num_classes=10,
    model_depth=10,
    shortcut_type="B",
    freeze_backbone=True,
)


class TopKPrediction(BaseModel):
    vertebra: str
    topk: List[Tuple[str, float]]


class PredictionResponse(BaseModel):
    predictions: List[TopKPrediction]
    unfound_vertebrae: List[str]


@app.get("/predict", response_model=PredictionResponse)
def predict_from_patient_id(
    patient_id: str = Query(..., description="Subdirectory under DICOM_DATA_DIR"),
    k: int = Query(2, description="Number of top-k predictions to return per vertebra"),
):
    """
    Predict vertebral injury classes using a patient's DICOM series stored under DICOM_DATA_DIR.

    Args:
        patient_id (str): Folder name inside DICOM_DATA_DIR containing the DICOM series.
        k (int): Number of top predictions per vertebra.

    Returns:
        JSON with predictions and list of vertebrae not found.
    """
    dicom_path = DICOM_DATA_DIR / patient_id
    if not dicom_path.exists() or not dicom_path.is_dir():
        return JSONResponse(
            status_code=400, content={"error": f"Directory '{dicom_path}' does not exist."}
        )

    try:
        predictions, unfound = detector.predict(patient_dicom_path=dicom_path, k=k)
        return PredictionResponse(
            predictions=[
                TopKPrediction(vertebra=p["vertebra"], topk=p["topk"]) for p in predictions
            ],
            unfound_vertebrae=sorted(unfound),
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

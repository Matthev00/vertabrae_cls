from pathlib import Path
from random import sample
from typing import List, Tuple

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import CLS_MODEL_DIR, DICOM_DATA_DIR
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


@app.get("/predict_fake", response_model=PredictionResponse)
def predict_fake_from_patient_id(
    patient_id: str = Query(..., description="Subdirectory under DICOM_DATA_DIR"),
    k: int = Query(2, description="Number of top-k predictions to return per vertebra"),
):
    from random import choice, randint, uniform

    vertebrae = [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "Th1",
        "Th2",
        "Th3",
        "Th4",
        "Th5",
        "Th6",
        "Th7",
        "Th8",
        "Th9",
        "Th10",
        "Th11",
        "Th12",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
    ]
    class_names = ["B1", "B2", "B3", "H"]

    start_idx = randint(7, len(vertebrae) - 4)
    unfound = set(sample(vertebrae, k=randint(0, 2)))
    found = [v for v in vertebrae if v not in unfound]
    injured_vertebrae = vertebrae[start_idx : start_idx + 3]

    predictions = []
    for v in found:
        if v in injured_vertebrae:
            topk = [choice(["B1", "B2", "B3"])] + ["H"]
            result = [
                (cls, round(uniform(0.6, 0.9), 6) if cls != "H" else round(uniform(0.1, 0.3), 6))
                for cls in topk
            ]
        else:
            topk = ["H"] + [choice(["B1", "B2", "B3"])]
            result = [
                (cls, round(uniform(0.7, 0.9), 6) if cls == "H" else round(uniform(0.1, 0.3), 6))
                for cls in topk
            ]
        predictions.append({"vertebra": v, "topk": result})

    return PredictionResponse(
        predictions=[TopKPrediction(vertebra=p["vertebra"], topk=p["topk"]) for p in predictions],
        unfound_vertebrae=sorted(unfound),
    )

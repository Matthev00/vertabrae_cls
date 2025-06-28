from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
### TEMP
DATA_DIR = Path("/media/mateusz/T7/Praca_Inzynierska/data")
###
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
DICOM_DATA_DIR = Path("/media/mateusz/T7/Praca_Inzynierska/Anonimized/")
RAPORT_FILE_PATH = INTERIM_DATA_DIR / "extracted_data.csv"
LABELS_FILE_PATH = PROCESSED_DATA_DIR / "labels.csv"
CLASS_NAMES_FILE_PATH = PROCESSED_DATA_DIR / "class_names.txt"
TENSOR_DIR = PROCESSED_DATA_DIR / "tensors"

MODELS_DIR = PROJ_ROOT / "models"
SEG_MODEL_DIR = MODELS_DIR / "monai_seg"
CLS_MODEL_DIR = MODELS_DIR / "cls"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# Constants
VERTEBRAE_MAP = {
    "L5": 18,
    "L4": 19,
    "L3": 20,
    "L2": 21,
    "L1": 22,
    "Th12": 23,
    "Th11": 24,
    "Th10": 25,
    "Th9": 26,
    "Th8": 27,
    "Th7": 28,
    "Th6": 29,
    "Th5": 30,
    "Th4": 31,
    "Th3": 32,
    "Th2": 33,
    "Th1": 34,
    "C7": 35,
    "C6": 36,
    "C5": 37,
    "C4": 38,
    "C3": 39,
    "C2": 40,
    "C1": 41,
}

# Model config
IS_FULL_RESOLUTION = True
DEVICE = "cpu"
TARGET_TENSOR_SIZE = (64, 64, 64)

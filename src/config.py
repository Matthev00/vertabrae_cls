from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
SEG_MODEL_DIR = MODELS_DIR / "monai_seg"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# Constants
VERTABRAE_MAP = {
    "L5": 18,
    "L4": 19,
    "L3": 20,
    "L2": 21,
    "L1": 22,
    "T12": 23,
    "T11": 24,
    "T10": 25,
    "T9": 26,
    "T8": 27,
    "T7": 28,
    "T6": 29,
    "T5": 30,
    "T4": 31,
    "T3": 32,
    "T2": 33,
    "T1": 34,
    "C7": 35,
    "C6": 36,
    "C5": 37,
    "C4": 38,
    "C3": 39,
    "C2": 40,
    "C1": 41,
}

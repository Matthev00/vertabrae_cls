import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def get_logger(name: str, log_path: Path = Path("logs/log.txt")) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def set_seed(seed: int) -> None:
    """
    Set seed for full reproducibility in Python, NumPy, PyTorch (CPU & GPU) and CUDNN.

    Args:
        seed (int): Random seed to set.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False


DETAILED_TO_MAIN_MAPPING = {
    "A0": "A",
    "A1": "A",
    "A2": "A",
    "A3": "A",
    "A4": "A",
    "B1": "B",
    "B2": "B",
    "B3": "B",
    "C": "C",
    "H": "H",
}

MAIN_CLASSES = ["A", "B", "C", "H"]

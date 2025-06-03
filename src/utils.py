import logging
import os
import random

import numpy as np
import torch


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(name)


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

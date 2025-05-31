import os
import random

import numpy as np
import torch


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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

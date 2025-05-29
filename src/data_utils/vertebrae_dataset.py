from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from src.config import CLASS_NAMES_FILE_PATH


class VertebraeDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, tensor_dir: Path, transform: Optional[Compose] = None
    ) -> None:
        """
        VertebraeDataset is a PyTorch Dataset for loading vertebrae data.

        Args:
            df (pd.DataFrame): DataFrame containing metadata for the dataset.
            tensor_dir (Path): Directory where tensor files are stored.
            transform (Optional[Compose]): Optional transformations to apply to the data.
        """
        self.df = df.reset_index(drop=True)
        self.tensor_dir = tensor_dir
        self.transform = transform
        self.class_mapping = self._load_class_mapping(CLASS_NAMES_FILE_PATH)

    def __len__(self):
        return len(self.df)

    def _load_class_mapping(self, filepath: Path) -> dict[str, int]:
        """
        Load class mapping from a text file.

        Args:
            filepath (Path): Path to the text file containing class names.

        Returns:
            dict[str, int]: A dictionary mapping class names to indices.
        """
        with open(filepath) as f:
            classes = [line.strip() for line in f if line.strip()]
        return {cls_name: idx for idx, cls_name in enumerate(classes)}

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        tensor_name = str(row["tensor_path"])
        if tensor_name.startswith("augmented/"):
            tensor_name = tensor_name.replace("augmented/", "")
            tensor_dir = self.tensor_dir / "augmented"
        else:
            tensor_dir = self.tensor_dir
        tensor_path = tensor_dir / f"{int(tensor_name):05d}.pt"
        tensor = torch.load(tensor_path, weights_only=False).squeeze(0)
        label = self.class_mapping[row["injury_type"]]

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label

from pathlib import Path
from typing import Optional
import torch

import pandas as pd
from monai.transforms import (
    NormalizeIntensity,
    Rand3DElastic,
    RandAdjustContrast,
    RandAffine,
    RandRotate,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.data_utils.vertebrae_dataset import VertebraeDataset


def read_labels_file(labels_file_path: Path) -> pd.DataFrame:
    """
    Reads the labels file and returns a DataFrame.

    Args:
        labels_file_path (Path): Path to the labels file.

    Returns:
        pd.DataFrame: DataFrame containing the labels.
    """
    df = pd.read_csv(labels_file_path)
    return df


def default_train_transforms() -> Compose:
    """
    Returns default transformations for training data.

    Returns:
        Compose: A composition of transformations for training data.
    """
    return Compose(
        [
            NormalizeIntensity(channel_wise=True),
            Rand3DElastic(
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                prob=1.0,
                spatial_size=tuple(torch.Size([64, 64, 64])),
            ),
            RandAdjustContrast(prob=0.5, gamma=(0.8, 1.2)),
            RandAffine(
                prob=0.5,
                rotate_range=(0.1, 0.1, 0.1),
                translate_range=(0.05, 0.05, 0.05),
                scale_range=(0.9, 1.1),
            ),
            RandRotate(
                prob=0.5,
                range_x=(-0.1, 0.1),
                range_y=(-0.1, 0.1),
                range_z=(-0.1, 0.1),
            ),
        ]
    )


def default_val_transforms() -> Compose:
    """
    Returns default transformations for validation data.

    Returns:
        Compose: A composition of transformations for validation data.
    """
    return Compose(
        [
            NormalizeIntensity(channel_wise=True),
        ]
    )


def get_dataloders(
    labels_file_path: Path,
    tensor_dir: Path,
    train_transforms: Optional[Compose] = None,
    val_transforms: Optional[Compose] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True,
    train_split: float = 0.8,
    balance_train: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Creates DataLoaders for training and validation datasets.

    Args:
        labels_file_path (Path): Path to the labels file.
        tensor_dir (Path): Directory where tensor files are stored.
        train_transforms (Optional[Compose]): Transformations for training data.
        val_transforms (Optional[Compose]): Transformations for validation data.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker threads for DataLoaders.
        shuffle_train (bool): Whether to shuffle training data.
        train_split (float): Proportion of data to use for training.
        balance_train (bool): Whether to balance the training dataset.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    df = read_labels_file(labels_file_path)
    # train_df, val_df = train_test_split(
    #     df, train_size=train_split, stratify=df["injury_type"], random_state=42
    # )
    # Podmienić
    train_df, val_df = train_test_split(
        df, train_size=train_split, random_state=42
    )

    if train_transforms is None:
        train_transforms = default_train_transforms()
    if val_transforms is None:
        val_transforms = default_val_transforms()

    if balance_train:
        return "Not implemented yet"

    train_dataset = VertebraeDataset(
        df=train_df,
        tensor_dir=tensor_dir,
        transform=train_transforms,
    )
    val_dataset = VertebraeDataset(
        df=val_df,
        tensor_dir=tensor_dir,
        transform=val_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    labels_file = Path("data/processed/labels.csv")
    tensor_directory = Path("data/processed/tensors")
    train_loader, val_loader = get_dataloders(
        labels_file_path=labels_file,
        tensor_dir=tensor_directory,
        batch_size=2,
        num_workers=1,
    )
    print(f"Train Loader: {len(train_loader)} batches")
    print(f"Validation Loader: {len(val_loader)} batches")
    sample = next(iter(train_loader))
    print(f"Sample batch size: {sample[0].shape}, Labels: {sample[1]}")
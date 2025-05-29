import random
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
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

AUGMENT_TRANSFORMS = [
    Rand3DElastic(
        sigma_range=(5, 8),
        magnitude_range=(100, 200),
        prob=1.0,
        spatial_size=(64, 64, 64),
    ),
    RandAdjustContrast(prob=1.0, gamma=(0.8, 1.2)),
    RandAffine(
        prob=1.0,
        rotate_range=(0.1, 0.1, 0.1),
        translate_range=(0.05, 0.05, 0.05),
        scale_range=(0.9, 1.1),
    ),
    RandRotate(
        prob=1.0,
        range_x=(-0.1, 0.1),
        range_y=(-0.1, 0.1),
        range_z=(-0.1, 0.1),
    ),
    Compose(
        [
            Rand3DElastic(
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                prob=0.5,
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
    ),
]


def get_max_index(self, dir) -> int:
    """
    Gets max existing tensor index.

    Returns:
        int: max existing tensor index.
    """
    existing_files = list(dir.glob("*.pt"))
    if existing_files:
        max_index = max(int(f.stem) for f in existing_files if f.stem.isdigit())
        return max_index
    return 0


def balance_dataframe_with_augmentation(
    df: pd.DataFrame,
    tensor_dir: Path,
) -> pd.DataFrame:
    """
    Balances the dataset by augmenting samples of underrepresented classes.
    Creates augmented tensors and saves them in a new directory.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset labels.
        tensor_dir (Path): Directory where tensor files are stored.

    Returns:
        pd.DataFrame: DataFrame with balanced dataset.
    """

    augmented_labels_path = tensor_dir.parent / "labels_augmented.csv"
    augmented_tensor_dir = tensor_dir / "augmented"
    augmented_tensor_dir.mkdir(parents=True, exist_ok=True)

    if augmented_labels_path.exists():
        print("[INFO] Augmented dataset found. Loading from disk.")
        return pd.read_csv(augmented_labels_path)

    class_counts = df["injury_type"].value_counts()
    max_count = class_counts.max()

    balanced_rows = [df]

    class_number = 0

    for cls, count in class_counts.items():
        deficit = max_count - count
        if deficit <= 0:
            continue

        samples_to_augment = df[df["injury_type"] == cls]
        augment_factor = int(np.ceil(deficit / len(samples_to_augment)))

        new_rows = []
        for i in range(min(deficit, augment_factor * len(samples_to_augment))):
            row = samples_to_augment.iloc[i % len(samples_to_augment)].copy()
            orig_tensor_path = tensor_dir / f"{row['tensor_path']:05d}.pt"

            tensor = torch.load(orig_tensor_path, weights_only=False).squeeze(0)

            transforms = random.choice(AUGMENT_TRANSFORMS)
            augmented_tensor = transforms(deepcopy(tensor))

            new_tensor_id = f"{class_number + i:05d}"
            new_tensor_path = augmented_tensor_dir / f"{new_tensor_id}.pt"
            torch.save(augmented_tensor.unsqueeze(0), new_tensor_path)

            row["tensor_path"] = f"augmented/{new_tensor_id}"
            new_rows.append(row)

        balanced_rows.append(pd.DataFrame(new_rows))
        class_number += len(new_rows)

    df_balanced = pd.concat(balanced_rows, ignore_index=True)
    df_balanced.to_csv(augmented_labels_path, index=False)

    return df_balanced


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
    train_df, val_df = train_test_split(df, train_size=train_split, random_state=42)

    if train_transforms is None:
        train_transforms = default_train_transforms()
    if val_transforms is None:
        val_transforms = default_val_transforms()

    if balance_train:
        train_df = balance_dataframe_with_augmentation(
            df=train_df,
            tensor_dir=tensor_dir,
        )

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
        balance_train=False,
    )
    print(f"Train Loader: {len(train_loader)} batches")
    print(f"Validation Loader: {len(val_loader)} batches")
    sample = next(iter(train_loader))
    print(f"Sample batch size: {sample[0].shape}, Labels: {sample[1]}")

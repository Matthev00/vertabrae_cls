from pathlib import Path
from typing import Optional, Literal

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

from src.data_utils.dataset_balancer import ProportionalDatasetBalancer, DatasetBalancer
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
                sigma_range=(2, 4),
                magnitude_range=(2, 8),
                prob=0.5,
                rotate_range=(0, 0, 0),
                scale_range=(0, 0, 0),
                translate_range=(0, 0, 0),
                padding_mode="border",
            ),
            RandAdjustContrast(prob=0.5, gamma=(0.8, 1.2)),
            RandAffine(
                prob=0.5,
                rotate_range=(0.1, 0.1, 0.1),
                translate_range=(0.05, 0.05, 0.05),
                scale_range=(0.9, 1.1),
                padding_mode="border",
            ),
            RandRotate(
                prob=0.5,
                range_x=(-0.1, 0.1),
                range_y=(-0.1, 0.1),
                range_z=(-0.1, 0.1),
                padding_mode="border",
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


def balance_dataframe(
    df: pd.DataFrame, tensor_dir: Path, balancer_type: Literal["base", "proportional"]
) -> pd.DataFrame:
    """
    Balances the dataset by augmenting samples of underrepresented classes.
    Creates augmented tensors and saves them in a new directory.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset labels.
        tensor_dir (Path): Directory where tensor files are stored.
        balancer_type Literal["base", "proportional"]: Type of data balncer.

    Returns:
        pd.DataFrame: DataFrame with balanced dataset.
    """
    augmented_labels_path = tensor_dir.parent / "labels_augmented.csv"
    augmented_tensor_dir = tensor_dir / "augmented"
    augmented_tensor_dir.mkdir(parents=True, exist_ok=True)

    if balancer_type == "base":
        balancer = DatasetBalancer(tensor_dir=tensor_dir)
        balanced_df = balancer.balance_dataframe_with_augmentation(df)
    elif balancer_type == "proportional":
        balancer = ProportionalDatasetBalancer(tensor_dir=tensor_dir)
        balanced_df = balancer.balance_dataframe_with_augmentation(df, k=3)
    balanced_df.to_csv(augmented_labels_path, index=False)
    return balanced_df


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
    main_classes: bool = False,
    balancer_type: Literal["base", "proportional"] = "base",
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
        main_classes (bool): If True, use only main classes for binary classification.
        balancer_type Literal["base", "proportional"]: Type of data balncer.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    df = read_labels_file(labels_file_path)
    train_df, val_df = train_test_split(
        df, train_size=train_split, stratify=df["injury_type"], random_state=42
    )

    if train_transforms is None:
        if balance_train:
            train_transforms = default_val_transforms()
        else:
            train_transforms = default_train_transforms()

    if val_transforms is None:
        val_transforms = default_val_transforms()

    if balance_train:
        train_df = balance_dataframe(
            df=train_df,
            tensor_dir=tensor_dir,
            balancer_type=balancer_type,
        )

    train_dataset = VertebraeDataset(
        df=train_df, tensor_dir=tensor_dir, transform=train_transforms, main_classes=main_classes
    )
    val_dataset = VertebraeDataset(
        df=val_df, tensor_dir=tensor_dir, transform=val_transforms, main_classes=main_classes
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

import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.transforms import Rand3DElastic, RandAdjustContrast, RandAffine, RandRotate
from torchvision.transforms import Compose


class DatasetBalancer:
    def __init__(self, tensor_dir: Path) -> None:
        self.tensor_dir = tensor_dir
        self.augment_transforms = [
            Rand3DElastic(
                sigma_range=(2, 4),
                magnitude_range=(2, 8),
                prob=1,
                rotate_range=(0, 0, 0),
                scale_range=(0, 0, 0),
                translate_range=(0, 0, 0),
                padding_mode="border",
            ),
            RandAdjustContrast(prob=1.0, gamma=(0.8, 1.2)),
            RandAffine(
                prob=1.0,
                rotate_range=(0.1, 0.1, 0.1),
                translate_range=(0.05, 0.05, 0.05),
                scale_range=(0.9, 1.1),
                padding_mode="border",
            ),
            RandRotate(
                prob=1.0,
                range_x=(-0.1, 0.1),
                range_y=(-0.1, 0.1),
                range_z=(-0.1, 0.1),
                padding_mode="border",
            ),
            Compose(
                [
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

    def balance_dataframe_with_augmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balances the dataset by augmenting samples of underrepresented classes.
        Creates augmented tensors and saves them in a new directory.

        Args:
            df (pd.DataFrame): DataFrame containing the dataset labels.

        Returns:
            pd.DataFrame: DataFrame with balanced dataset.
        """

        augmented_tensor_dir = self.tensor_dir / "augmented"
        augmented_tensor_dir.mkdir(parents=True, exist_ok=True)
        balanced_rows = [df]
        df = df[df["injury_type"] != "H"]

        class_counts = df["injury_type"].value_counts()
        max_count = class_counts.max()

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
                orig_tensor_path = self.tensor_dir / f"{row['tensor_path']:05d}.pt"

                tensor = torch.load(orig_tensor_path, weights_only=False).squeeze(0)

                transforms = random.choice(self.augment_transforms)
                augmented_tensor = transforms(deepcopy(tensor))

                new_tensor_id = f"{class_number + i:05d}"
                new_tensor_path = augmented_tensor_dir / f"{new_tensor_id}.pt"
                torch.save(augmented_tensor.unsqueeze(0), new_tensor_path)

                row["tensor_path"] = f"augmented/{new_tensor_id}"
                new_rows.append(row)

            balanced_rows.append(pd.DataFrame(new_rows))
            class_number += len(new_rows)

        return pd.concat(balanced_rows, ignore_index=True)


class ProportionalDatasetBalancer(DatasetBalancer):
    def balance_dataframe_with_augmentation(self, df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        """
        Balances the dataset proportionally by augmenting samples in each class.
        Each class will have its size multiplied by `k`.

        Args:
            df (pd.DataFrame): DataFrame containing the dataset labels.
            k (int): Multiplication factor for the number of samples in each class.

        Returns:
            pd.DataFrame: DataFrame with proportionally balanced dataset.
        """
        augmented_tensor_dir = self.tensor_dir / "augmented"
        augmented_tensor_dir.mkdir(parents=True, exist_ok=True)
        balanced_rows = [df]
        df = df[df["injury_type"] != "H"]
        class_counts = df["injury_type"].value_counts()

        class_number = 0

        for cls, count in class_counts.items():
            target_count = count * k
            deficit = target_count - count

            samples_to_augment = df[df["injury_type"] == cls]

            new_rows = []
            for i in range(deficit):
                row = samples_to_augment.iloc[i % len(samples_to_augment)].copy()
                orig_tensor_path = self.tensor_dir / f"{row['tensor_path']:05d}.pt"

                tensor = torch.load(orig_tensor_path, weights_only=False).squeeze(0)

                transforms = random.choice(self.augment_transforms)
                augmented_tensor = transforms(deepcopy(tensor))

                new_tensor_id = f"{class_number + i:05d}"
                new_tensor_path = augmented_tensor_dir / f"{new_tensor_id}.pt"
                torch.save(augmented_tensor.unsqueeze(0), new_tensor_path)

                row["tensor_path"] = f"augmented/{new_tensor_id}"
                new_rows.append(row)

            balanced_rows.append(pd.DataFrame(new_rows))
            class_number += len(new_rows)

        return pd.concat(balanced_rows, ignore_index=True)

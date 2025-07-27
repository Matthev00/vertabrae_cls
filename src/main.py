import os
from collections import Counter

import torch

from config import LABELS_FILE_PATH, TENSOR_DIR
from src.data_utils.dataloader import get_dataloders
from src.modeling.model_factory import create_model
from src.utils import set_seed


def main():
    set_seed(42)
    num_workers = os.cpu_count()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {"model_depth": 18, "freeze_backbone": False}
    model = create_model(model_type="monai", num_classes=10, device=device, **config)

    train_dataloader, val_dataloader = get_dataloders(
        labels_file_path=LABELS_FILE_PATH,
        tensor_dir=TENSOR_DIR,
        batch_size=4,
        balance_train=True,
        num_workers=num_workers,
        train_split=0.7,
        binary_class=True,
        balancer_type="proportional",
    )
    class_counts = Counter()
    for _, y in val_dataloader:
        class_counts.update(y.tolist())

    print("Class distribution in validation dataset:")
    for cls, count in class_counts.items():
        print(f"Class {cls}: {count} examples")

    print(len(train_dataloader.dataset), "examples in training dataset")
    print(len(val_dataloader.dataset), "examples in validation dataset")

    class_counts = Counter()
    for _, y in train_dataloader:
        class_counts.update(y.tolist())
    print("Class distribution in training dataset:")
    for cls, count in class_counts.items():
        print(f"Class {cls}: {count} examples")


if __name__ == "__main__":
    main()

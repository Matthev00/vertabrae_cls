import os

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
    model = create_model(model_type="med3d", num_classes=10, device=device, **config)

    train_dataloader, _ = get_dataloders(
        labels_file_path=LABELS_FILE_PATH,
        tensor_dir=TENSOR_DIR,
        batch_size=4,
        balance_train=True,
        num_workers=num_workers,
    )
    X, y = next(iter(train_dataloader))
    X = X.to(device)
    y_pred = model.predict()
    print(f"Pred: {y_pred}, label: {y}")


if __name__ == "__main__":
    main()

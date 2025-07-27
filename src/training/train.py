import argparse
import os
from typing import Optional

import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

import wandb
from src.config import (
    CLASS_NAMES_FILE_PATH_BINARY,
    LABELS_FILE_PATH,
    MODELS_DIR,
    TENSOR_DIR,
)
from src.data_utils.dataloader import get_dataloders
from src.modeling.model_factory import create_model
from src.training.engine import Trainer
from src.utils import set_seed


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    sched_cfg = config.get("scheduler")
    if sched_cfg is None or sched_cfg.get("type") is None:
        return None

    sched_type = sched_cfg["type"]
    kwargs = {k: v for k, v in sched_cfg.items() if k != "type"}

    if sched_type == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, **kwargs)
    elif sched_type == "StepLR":
        return StepLR(optimizer, **kwargs)
    elif sched_type == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")


def train(config: dict = None):
    """
    Training entry point compatible with both WandB Sweep and CLI.

    Args:
        config (dict, optional): Configuration dictionary. If None, will be read from CLI.
    """
    if config is None:
        raise ValueError("Configuration dictionary must be provided when using wandb sweep.")
    set_seed(42)
    NUM_WORKERS = os.cpu_count()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    run = wandb.init(config=config, project="Vertebrae Classifier")
    config = wandb.config

    train_loader, val_loader = get_dataloders(
        labels_file_path=LABELS_FILE_PATH,
        tensor_dir=TENSOR_DIR,
        train_transforms=None,
        val_transforms=None,
        batch_size=config["batch_size"],
        num_workers=0,
        balance_train=config.get("balance_train", False),
        train_split=0.7,
        binary_class=True,
        balancer_type=config.get("balancer_type", "base"),
    )
    with open(CLASS_NAMES_FILE_PATH_BINARY) as f:
        class_names = [line.strip() for line in f if line.strip()]

    model = create_model(
        model_type=config["model_type"],
        num_classes=len(class_names),
        model_depth=config.get("model_depth", 18),
        shortcut_type=config.get("shortcut_type", "B"),
        freeze_backbone=config.get("freeze_backbone", True),
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = build_scheduler(optimizer=optimizer, config=config)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        run_name=run.name,
        scheduler=scheduler,
        early_stopping=config["early_stopping"],
        early_stopping_patience=config["early_stopping_patience"],
        class_names=class_names,
        max_epochs=config["max_epochs"],
        log_wandb=True,
        save_dir=MODELS_DIR / "cls",
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D classifier with config file")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)

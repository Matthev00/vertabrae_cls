from pathlib import Path

import wandb

from src.training.train import train

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_recall", "goal": "maximize"},
    "parameters": {
        "model_type": {"values": ["monai"]},
        "freeze_backbone": {"values": [True, False]},
        "batch_size": {"values": [8, 16, 32, 64, 128, 256]},
        "lr": {"min": 1e-4, "max": 1e-1},
        "scheduler_type": {"values": ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]},
        "step_size": {"values": [5, 10]},
        "gamma": {"values": [0.1, 0.5]},
        "T_max": {"values": [20, 50]},
        "eta_min": {"values": [1e-5, 1e-4]},
        "max_epochs": {"value": 70},
        "weight_decay": {"values": [0.0, 1e-5, 1e-3, 5e-2]},
        "balance_train": {"values": [True, False]},
        "early_stopping_patience": {"value": 20},
        "balancer_type": {"values": ["proportional", "base"]},
    },
}


def sweep_train():
    wandb.init(project="Vertebrae Classifier")
    train(dict(wandb.config))


if __name__ == "__main__":
    sweep_file = Path("sweep_id.txt")

    if sweep_file.exists():
        sweep_id = sweep_file.read_text().strip()
        print(f"Resuming existing sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(sweep_config, project="Vertebrae Classifier - Binary")
        sweep_file.write_text(sweep_id)
        print(f"Created new sweep: {sweep_id}")

    wandb.agent(sweep_id, function=sweep_train, count=100)

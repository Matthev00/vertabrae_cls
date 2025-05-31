from src.training.train import train
import wandb

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_balanced_acc", "goal": "maximize"},
    "parameters": {
        "model_type": {"values": ["med3d", "monai", "base"]},
        "model_depth": {"values": [10, 18, 34, 50, 101, 152, 200]},
        "freeze_backbone": {"values": [True, False]},
        "batch_size": {"values": [8, 16, 32]},
        "lr": {"min": 1e-5, "max": 1e-2},
        "scheduler_type": {"values": ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]},
        "step_size": {"values": [5, 10]},
        "gamma": {"values": [0.1, 0.5]},
        "T_max": {"values": [20, 50]},
        "eta_min": {"values": [1e-5, 1e-4]},
        "max_epochs": {"value": 50},
        "weight_decay": {"values": [0.0, 1e-5, 1e-4]},
        "shortcut_type": {"value": "B"},
        "balance_train": {"value": True},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="vertebrae-classifier")
    wandb.agent(sweep_id, function=train, count=100)

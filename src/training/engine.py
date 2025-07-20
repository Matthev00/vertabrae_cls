import copy
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCohenKappa,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassStatScores,
)

import wandb
from src.config import MODELS_DIR


class Trainer:
    """
    Trainer class for supervised classification of 3D data.

    Features:
    - Logs metrics to Weights & Biases (optional).
    - Supports learning rate scheduler.
    - Saves best model checkpoint locally.
    - Logs final confusion matrix for best model only.

    Args:
        model (nn.Module): PyTorch model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        device (torch.device): Target device (CPU/GPU).
        run_name (str): Name for saving best model and wandb run.
        scheduler (Optional[_LRScheduler]): Optional learning rate scheduler.
        save_dir (Path): Directory for saving checkpoints.
        log_wandb (bool): Whether to log metrics to Weights & Biases.
        class_names (Optional[list[str]]): Class labels for confusion matrix.
        max_epochs (int): Number of training epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.CrossEntropyLoss,
        device: torch.device,
        run_name: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        save_dir: Path = MODELS_DIR / "cls",
        log_wandb: bool = True,
        class_names: Optional[list[str]] = None,
        max_epochs: int = 50,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_wandb = log_wandb
        self.max_epochs = max_epochs
        self.run_name = run_name
        self.class_names = class_names
        num_classes = len(self.class_names) if class_names is not None else 9

        average = "macro"
        self.train_metrics: dict[str, MulticlassStatScores] = {
            "acc": MulticlassAccuracy(num_classes=num_classes).to(device),
            "balanced_acc": MulticlassAccuracy(num_classes=num_classes, average=average),
            "mcc": MulticlassMatthewsCorrCoef(num_classes=num_classes).to(device),
            "kappa": MulticlassCohenKappa(num_classes=num_classes).to(device),
            "jaccard": MulticlassJaccardIndex(num_classes=num_classes, average=average).to(device),
            "precision": MulticlassPrecision(num_classes=num_classes, average=average).to(device),
            "recall": MulticlassRecall(num_classes=num_classes, average=average).to(device),
            "f1": MulticlassF1Score(num_classes=num_classes, average=average).to(device),
        }
        self.val_metrics: dict[str, MulticlassStatScores] = {
            k: copy.deepcopy(v).to(device) for k, v in self.train_metrics.items()
        }

        self.best_val_recall = 0.0
        self.best_model_path = self.save_dir / f"{run_name}_best.pt"
        self._val_preds: list[int] = []
        self._val_targets: list[int] = []

        if self.log_wandb:
            wandb.watch(self.model)

    def train(self):
        """
        Main training loop.

        For each epoch:
        - Runs training and validation
        - Logs metrics to wandb (if enabled)
        - Saves best model (lowest validation loss)
        - Stores predictions and labels to log confusion matrix later
        """
        epochs_no_improve = 0
        for epoch in range(1, self.max_epochs + 1):
            train_loss, train_metrics = self._train_one_epoch()
            val_loss, val_preds, val_targets, val_metrics = self._validate()

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if self.log_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        **train_metrics,
                        **val_metrics,
                    }
                )

            val_recall = val_metrics.get("val_recall", 0.0)
            if val_recall > self.best_val_recall:
                self.best_val_recall = val_recall
                torch.save(self.model.state_dict(), self.best_model_path)
                self._val_preds = val_preds
                self._val_targets = val_targets
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if self.early_stopping and epochs_no_improve >= self.early_stopping_patience:
                print(
                    f"⏹️ Early stopping at epoch {epoch} (no improvement for {self.early_stopping_patience} epochs)"
                )
                break

        if self.log_wandb and self._val_preds and self._val_targets:
            if not self.class_names:
                self.class_names = [str(i) for i in sorted(set(self._val_targets))]

            wandb.log(
                {
                    "val_confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=self._val_targets,
                        preds=self._val_preds,
                        class_names=self.class_names,
                    )
                }
            )

        print(f"✅ Best model saved at: {self.best_model_path}")

    def _train_one_epoch(self) -> tuple[float, dict[str, torch.Tensor]]:
        """
        Runs one epoch of training.

        Returns:
            avg_loss (float): Average training loss.
            metrics_out (dict[str, torch.Tensor]): Dictionary of computed validation metrics,
                including accuracy, precision, recall, F1-score, MCC, Cohen's kappa,
                Jaccard index and balanced accuracy (macro-averaged).
        """
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for metric in self.train_metrics.values():
            metric.reset()

        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_targets.append(y)

        preds_all = torch.cat(all_preds)
        targets_all = torch.cat(all_targets)

        for metric in self.train_metrics.values():
            preds = preds_all.to(metric.device)
            targets = targets_all.to(metric.device)
            metric.update(preds, targets)

        avg_loss = total_loss / len(self.train_loader.dataset)
        metrics_out = {f"train_{k}": m.compute().item() for k, m in self.train_metrics.items()}
        return avg_loss, metrics_out

    @torch.inference_mode()
    def _validate(self) -> tuple[float, list[int], list[int], dict[str, torch.Tensor]]:
        """
        Runs validation on the current model.

        Returns:
            avg_loss (float): Average validation loss over the validation dataset.
            all_preds (list[int]): Flattened list of predicted class indices.
            all_targets (list[int]): Flattened list of ground truth class indices.
            metrics_out (dict[str, torch.Tensor]): Dictionary of computed validation metrics,
                including accuracy, precision, recall, F1-score, MCC, Cohen's kappa,
                Jaccard index and balanced accuracy (macro-averaged).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for metric in self.val_metrics.values():
            metric.reset()

        for X, y in self.val_loader:
            X, y = X.to(self.device), y.to(self.device)
            outputs = self.model(X)
            loss = self.criterion(outputs, y)

            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * X.size(0)
            all_preds.append(preds)
            all_targets.append(y)

        preds_all = torch.cat(all_preds)
        targets_all = torch.cat(all_targets)

        for metric in self.val_metrics.values():
            preds = preds_all.to(metric.device)
            targets = targets_all.to(metric.device)
            metric.update(preds, targets)

        avg_loss = total_loss / len(self.val_loader.dataset)
        metrics_out = {f"val_{k}": m.compute().item() for k, m in self.val_metrics.items()}

        return avg_loss, preds_all.cpu().tolist(), targets_all.cpu().tolist(), metrics_out

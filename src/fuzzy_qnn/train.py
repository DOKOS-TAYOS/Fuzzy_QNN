from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .evaluate import evaluate_model
from .model import FuzzyQuantumClassifier


@dataclass(slots=True)
class TrainingHistory:
    epoch: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float | None] = field(default_factory=list)
    val_accuracy: list[float | None] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    best_checkpoint: list[bool] = field(default_factory=list)

    def append(
        self,
        *,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: float | None,
        val_accuracy: float | None,
        learning_rate: float,
        epoch_seconds: float,
        best_checkpoint: bool,
    ) -> None:
        self.epoch.append(epoch)
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_accuracy)
        self.val_loss.append(val_loss)
        self.val_accuracy.append(val_accuracy)
        self.learning_rate.append(learning_rate)
        self.epoch_seconds.append(epoch_seconds)
        self.best_checkpoint.append(best_checkpoint)

    def to_dict(self) -> dict[str, list[int | float | bool | None]]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "learning_rate": self.learning_rate,
            "epoch_seconds": self.epoch_seconds,
            "best_checkpoint": self.best_checkpoint,
        }


@dataclass(slots=True)
class TrainResult:
    history: TrainingHistory
    best_epoch: int
    best_metric_name: str
    best_metric_value: float
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    train_seconds: float
    seconds_per_epoch_mean: float
    seconds_per_epoch_std: float


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    lambda_sigma: float,
    torch_device: torch.device,
    run_dir: Path,
    show_progress: bool = True,
    early_stopping: bool = True,
    patience: int = 20,
    min_delta: float = 1e-5,
) -> TrainResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = TrainingHistory()
    best_checkpoint_path = run_dir / "best_model.pt"
    last_checkpoint_path = run_dir / "model.pt"
    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_metric_name = "val_loss" if val_loader is not None else "train_loss"
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{epochs:03d}",
            leave=True,
            disable=not show_progress,
            file=sys.stdout,
        )

        for features, targets in progress:
            features = features.to(torch_device)
            targets = targets.to(torch_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = functional.cross_entropy(logits, targets)
            if lambda_sigma > 0.0 and isinstance(model, FuzzyQuantumClassifier):
                loss = loss + (lambda_sigma * model.fuzzy_block.sigma_penalty())
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            batch_size = features.size(0)
            running_loss += loss.item() * batch_size
            running_correct += int((predictions == targets).sum().item())
            running_total += batch_size
            progress.set_postfix(
                train_loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_loss = running_loss / running_total
        train_accuracy = running_correct / running_total
        val_loss: float | None = None
        val_accuracy: float | None = None
        if val_loader is not None:
            validation = evaluate_model(
                model=model,
                data_loader=val_loader,
                torch_device=torch_device,
            )
            val_loss = validation.loss
            val_accuracy = validation.accuracy
            metric_to_monitor = val_loss
        else:
            metric_to_monitor = train_loss

        is_best = metric_to_monitor < (best_metric - min_delta)
        if is_best:
            best_metric = metric_to_monitor
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint_path)
            bad_epochs = 0
        else:
            bad_epochs += 1

        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            epoch=epoch,
            train_loss=float(train_loss),
            train_accuracy=float(train_accuracy),
            val_loss=None if val_loss is None else float(val_loss),
            val_accuracy=None if val_accuracy is None else float(val_accuracy),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            epoch_seconds=float(epoch_seconds),
            best_checkpoint=is_best,
        )

        if show_progress:
            tqdm.write(
                " ".join(
                    [
                        f"epoch={epoch:03d}",
                        f"train_loss={train_loss:.4f}",
                        f"train_acc={train_accuracy:.4f}",
                        f"val_loss={'None' if val_loss is None else f'{val_loss:.4f}'}",
                        f"val_acc={'None' if val_accuracy is None else f'{val_accuracy:.4f}'}",
                        f"epoch_seconds={epoch_seconds:.2f}",
                        f"best={is_best}",
                    ]
                ),
                file=sys.stdout,
            )

        if early_stopping and bad_epochs >= patience:
            break

    torch.save(model.state_dict(), last_checkpoint_path)
    train_seconds = time.perf_counter() - train_start
    epoch_array = np.array(history.epoch_seconds, dtype=np.float64)
    return TrainResult(
        history=history,
        best_epoch=best_epoch or 1,
        best_metric_name=best_metric_name,
        best_metric_value=float(best_metric),
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        train_seconds=float(train_seconds),
        seconds_per_epoch_mean=float(epoch_array.mean()) if len(epoch_array) else math.nan,
        seconds_per_epoch_std=float(epoch_array.std()) if len(epoch_array) else math.nan,
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    torch_device: torch.device,
) -> None:
    state_dict = torch.load(Path(checkpoint_path), map_location=torch_device)
    model.load_state_dict(state_dict)

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from .model import FuzzyQuantumClassifier


@dataclass(slots=True)
class EvaluationResult:
    loss: float
    accuracy: float
    y_true: np.ndarray
    y_pred: np.ndarray
    probs: np.ndarray
    logits: np.ndarray


@dataclass(slots=True)
class TestMetrics:
    test_loss: float
    test_accuracy: float
    test_balanced_accuracy: float
    test_precision_macro: float
    test_recall_macro: float
    test_f1_macro: float
    test_f1_weighted: float
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]
    n_test_samples: int
    train_seconds: float | None = None
    seconds_per_epoch_mean: float | None = None
    seconds_per_epoch_std: float | None = None
    n_qubits: int | None = None
    n_rules: int | None = None
    n_quantum_layers: int | None = None
    torch_device: str | None = None
    quantum_device: str | None = None
    diff_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    torch_device: torch.device,
) -> EvaluationResult:
    model.eval()
    losses: list[float] = []
    true_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(torch_device)
            targets = targets.to(torch_device)
            logits = model(features)
            loss = functional.cross_entropy(logits, targets)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            losses.append(loss.item() * features.size(0))
            true_batches.append(targets.cpu().numpy())
            pred_batches.append(predictions.cpu().numpy())
            prob_batches.append(probs.cpu().numpy())
            logit_batches.append(logits.cpu().numpy())

    y_true = np.concatenate(true_batches)
    y_pred = np.concatenate(pred_batches)
    probs = np.concatenate(prob_batches)
    logits = np.concatenate(logit_batches)
    total = len(y_true)
    average_loss = float(sum(losses) / total)
    accuracy = float(accuracy_score(y_true, y_pred))
    return EvaluationResult(
        loss=average_loss,
        accuracy=accuracy,
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        logits=logits,
    )


def evaluate_test_set(
    model: torch.nn.Module,
    test_loader: DataLoader,
    torch_device: torch.device,
    class_names: list[str] | None = None,
    train_seconds: float | None = None,
    seconds_per_epoch_mean: float | None = None,
    seconds_per_epoch_std: float | None = None,
    n_qubits: int | None = None,
    n_rules: int | None = None,
    n_quantum_layers: int | None = None,
    quantum_device: str | None = None,
    diff_method: str | None = None,
) -> TestMetrics:
    evaluation = evaluate_model(model=model, data_loader=test_loader, torch_device=torch_device)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        evaluation.y_true,
        evaluation.y_pred,
        average="macro",
        zero_division=cast(Any, 0),
    )
    f1_weighted = f1_score(
        evaluation.y_true,
        evaluation.y_pred,
        average="weighted",
        zero_division=cast(Any, 0),
    )
    labels = class_names
    report = classification_report(
        evaluation.y_true,
        evaluation.y_pred,
        target_names=labels,
        zero_division=cast(Any, 0),
        output_dict=True,
    )
    return TestMetrics(
        test_loss=evaluation.loss,
        test_accuracy=evaluation.accuracy,
        test_balanced_accuracy=float(balanced_accuracy_score(evaluation.y_true, evaluation.y_pred)),
        test_precision_macro=float(precision_macro),
        test_recall_macro=float(recall_macro),
        test_f1_macro=float(f1_macro),
        test_f1_weighted=float(f1_weighted),
        confusion_matrix=confusion_matrix(evaluation.y_true, evaluation.y_pred).tolist(),
        classification_report=cast(dict[str, Any], report),
        n_test_samples=len(evaluation.y_true),
        train_seconds=train_seconds,
        seconds_per_epoch_mean=seconds_per_epoch_mean,
        seconds_per_epoch_std=seconds_per_epoch_std,
        n_qubits=n_qubits,
        n_rules=n_rules,
        n_quantum_layers=n_quantum_layers,
        torch_device=torch_device.type,
        quantum_device=quantum_device,
        diff_method=diff_method,
    )


def inspect_sample(
    model: FuzzyQuantumClassifier,
    x: torch.Tensor,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        outputs = model(x, return_intermediates=True)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)
        predicted_index = int(torch.argmax(probs, dim=1).item())
    return {
        "logits": logits.cpu().tolist(),
        "probs": probs.cpu().tolist(),
        "predicted_class": class_names[predicted_index] if class_names else predicted_index,
        "mu": outputs["mu"].cpu().tolist(),
        "alpha": outputs["alpha"].cpu().tolist(),
        "z": outputs["z"].cpu().tolist(),
    }

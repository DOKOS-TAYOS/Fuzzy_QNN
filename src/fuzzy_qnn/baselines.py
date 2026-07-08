from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier

from .config import ExperimentConfig
from .data import DataBundle
from .fuzzy import FuzzyBlock


def run_baselines(data_bundle: DataBundle, config: ExperimentConfig) -> list[dict[str, Any]]:
    return [
        _logistic_regression_baseline(data_bundle, config.seed),
        _mlp_baseline(data_bundle, config.seed),
        _fuzzy_classical_baseline(data_bundle, config),
    ]


def _logistic_regression_baseline(data_bundle: DataBundle, seed: int) -> dict[str, Any]:
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(data_bundle.x_train, data_bundle.y_train)
    probabilities = classifier.predict_proba(data_bundle.x_test)
    predictions = classifier.predict(data_bundle.x_test)
    return _baseline_metrics("logistic_regression", data_bundle.y_test, predictions, probabilities)


def _mlp_baseline(data_bundle: DataBundle, seed: int) -> dict[str, Any]:
    classifier = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=500,
        random_state=seed,
    )
    classifier.fit(data_bundle.x_train, data_bundle.y_train)
    probabilities = classifier.predict_proba(data_bundle.x_test)
    predictions = classifier.predict(data_bundle.x_test)
    return _baseline_metrics("mlp", data_bundle.y_test, predictions, probabilities)


def _fuzzy_classical_baseline(data_bundle: DataBundle, config: ExperimentConfig) -> dict[str, Any]:
    block = FuzzyBlock(
        d_in=data_bundle.metadata["d_in"],
        n_fuzzy_sets=config.model.n_fuzzy_sets,
        n_rules=config.model.n_rules,
        seed=config.seed,
    )
    with torch.no_grad():
        train_alpha = block(torch.tensor(data_bundle.x_train, dtype=torch.float32))[1].cpu().numpy()
        test_alpha = block(torch.tensor(data_bundle.x_test, dtype=torch.float32))[1].cpu().numpy()
    classifier = LogisticRegression(max_iter=1000, random_state=config.seed)
    classifier.fit(train_alpha, data_bundle.y_train)
    probabilities = classifier.predict_proba(test_alpha)
    predictions = classifier.predict(test_alpha)
    return _baseline_metrics("fuzzy_classical", data_bundle.y_test, predictions, probabilities)


def _baseline_metrics(
    model_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    return {
        "model_type": model_type,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": float(log_loss(y_true, probabilities)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

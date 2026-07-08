from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .config import ExperimentConfig
from .train import TrainingHistory, TrainResult
from .utils import ensure_directory, save_json


def create_run_dir(experiment_name: str, output_dir: str | Path, timestamp: str) -> Path:
    return ensure_directory(Path(output_dir) / experiment_name / timestamp)


def save_history(
    history: TrainingHistory,
    run_dir: Path,
    *,
    save_csv: bool,
    save_json_copy: bool,
) -> tuple[Path | None, Path | None]:
    history_csv_path = run_dir / "history.csv"
    history_json_path = run_dir / "history.json"
    if save_csv:
        frame = pd.DataFrame(history.to_dict())
        frame.to_csv(history_csv_path, index=False)
    if save_json_copy:
        save_json(history_json_path, history.to_dict())
    return (
        history_csv_path if save_csv else None,
        history_json_path if save_json_copy else None,
    )


def apply_output_retention_policy(
    train_result: TrainResult,
    *,
    save_best_model: bool,
    save_last_model: bool,
) -> None:
    if not save_best_model and train_result.best_checkpoint_path.exists():
        train_result.best_checkpoint_path.unlink()
    if not save_last_model and train_result.last_checkpoint_path.exists():
        train_result.last_checkpoint_path.unlink()


def plot_loss_curves(history: TrainingHistory, output_path: str | Path) -> None:
    epochs = history.epoch
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.train_loss, label="train_loss")
    if any(value is not None for value in history.val_loss):
        plt.plot(
            epochs,
            [value if value is not None else float("nan") for value in history.val_loss],
            label="val_loss",
        )
    plt.xlabel("epoch")
    plt.ylabel("cross entropy loss")
    plt.title("Train/validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_accuracy_curves(history: TrainingHistory, output_path: str | Path) -> None:
    epochs = history.epoch
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.train_accuracy, label="train_accuracy")
    if any(value is not None for value in history.val_accuracy):
        plt.plot(
            epochs,
            [value if value is not None else float("nan") for value in history.val_accuracy],
            label="val_accuracy",
        )
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Train/validation accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_experiment_summary(
    config: ExperimentConfig,
    metadata: dict[str, Any],
    runtime_info: dict[str, Any],
    run_dir: Path,
) -> None:
    print("Run setup")
    print("---------")
    _print_key_value("name", config.experiment_name)
    _print_key_value("seed", config.seed)
    _print_key_value("dataset", config.dataset.name)
    _print_key_value(
        "split",
        (
            f"train={metadata['train_fraction']:.0%}, "
            f"val={metadata['val_fraction']:.0%}, "
            f"test={metadata['test_fraction']:.0%}"
        ),
    )
    _print_key_value("n_train", metadata["n_train"])
    _print_key_value("n_val", metadata["n_val"])
    _print_key_value("n_test", metadata["n_test"])
    _print_key_value("d_in", metadata["d_in"])
    _print_key_value("n_classes", metadata["n_classes"])
    _print_key_value("n_rules", config.model.n_rules)
    _print_key_value("n_fuzzy_sets", config.model.n_fuzzy_sets)
    _print_key_value("n_quantum_layers", config.model.n_quantum_layers)
    _print_key_value("n_qubits", runtime_info["n_qubits"])
    _print_key_value("torch_device", runtime_info["torch_device"])
    _print_key_value("quantum_device", runtime_info["quantum_device"])
    _print_key_value("diff_method", runtime_info["diff_method"])
    _print_key_value("output_dir", run_dir)

    if (
        config.runtime.quantum_device == "lightning.gpu"
        and runtime_info["quantum_device"] != "lightning.gpu"
    ):
        print(
            "note: requested 'lightning.gpu' was not available. "
            f"Falling back to '{runtime_info['quantum_device']}' because require_gpu=false."
        )
    if config.runtime.torch_device == "cuda" and runtime_info["torch_device"] != "cuda":
        print("note: requested Torch CUDA was not available. Falling back to CPU.")


def print_final_metrics(metrics: dict[str, Any], run_dir: Path) -> None:
    print("Final test metrics")
    print("------------------")
    _print_key_value("test_loss", f"{metrics['test_loss']:.4f}")
    _print_key_value("test_accuracy", f"{metrics['test_accuracy']:.4f}")
    _print_key_value("test_balanced_accuracy", f"{metrics['test_balanced_accuracy']:.4f}")
    _print_key_value("test_f1_macro", f"{metrics['test_f1_macro']:.4f}")
    _print_key_value("test_f1_weighted", f"{metrics['test_f1_weighted']:.4f}")
    print()
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    report = metrics.get("classification_report")
    if isinstance(report, dict):
        print()
        print("Classification report:")
        print(_format_classification_report(report))
    print()
    print("Artifacts saved to:")
    print(run_dir)


def _print_key_value(key: str, value: Any) -> None:
    print(f"{key}: {value}")


def _format_classification_report(report: dict[str, Any]) -> str:
    ordered_labels = [
        label
        for label in report
        if label not in {"accuracy", "macro avg", "weighted avg"}
        and isinstance(report[label], dict)
    ]
    ordered_labels.extend(
        label for label in ("macro avg", "weighted avg") if isinstance(report.get(label), dict)
    )
    lines = [f"{'class':<18}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"]
    for label in ordered_labels:
        values = report[label]
        lines.append(
            f"{label:<18}"
            f"{float(values['precision']):>10.4f}"
            f"{float(values['recall']):>10.4f}"
            f"{float(values['f1-score']):>10.4f}"
            f"{float(values['support']):>10.0f}"
        )
    accuracy = report.get("accuracy")
    if accuracy is not None:
        lines.append("")
        lines.append(f"overall accuracy: {float(accuracy):.4f}")
    return "\n".join(lines)

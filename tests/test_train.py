from __future__ import annotations

from pathlib import Path

from fuzzy_qnn.cli import run_training_experiment
from fuzzy_qnn.config import (
    DatasetConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    OutputsConfig,
    RuntimeConfig,
    TrainingConfig,
)
from fuzzy_qnn.data import build_dataloaders
from fuzzy_qnn.device import resolve_runtime
from fuzzy_qnn.model import FuzzyQuantumClassifier
from fuzzy_qnn.train import train_model


def _build_smoke_config(output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="smoke_train",
        seed=1234,
        dataset=DatasetConfig(
            name="synthetic",
            test_size=0.2,
            val_size=0.2,
            scale="minmax",
            feature_reduction="none",
            d_in=4,
            n_samples=48,
            n_classes=2,
            n_informative=3,
            n_redundant=1,
            class_sep=1.0,
            flip_y=0.01,
        ),
        model=ModelConfig(n_fuzzy_sets=3, n_rules=4, n_quantum_layers=1),
        training=TrainingConfig(
            batch_size=8,
            epochs=1,
            learning_rate=0.01,
            weight_decay=0.0,
            lambda_sigma=0.0,
            early_stopping=False,
            patience=3,
            min_delta=1e-5,
            deterministic=False,
        ),
        runtime=RuntimeConfig(
            prefer_gpu=False,
            require_gpu=False,
            torch_device="cpu",
            quantum_device="default.qubit",
            diff_method="backprop",
            quantum_inputs_on_cpu=False,
        ),
        logging=LoggingConfig(
            show_progress=False,
            plot_loss=False,
            plot_accuracy=False,
            save_history_csv=True,
            save_history_json=True,
            save_stdout_log=False,
        ),
        outputs=OutputsConfig(
            output_dir=str(output_dir),
            save_best_model=True,
            save_last_model=True,
            save_rules=True,
            save_fuzzy_parameters=True,
        ),
    )


def test_training_loop_runs_one_epoch_and_creates_checkpoints(temp_dir: Path) -> None:
    config = _build_smoke_config(temp_dir)
    data_bundle = build_dataloaders(config)
    runtime = resolve_runtime(
        config.runtime,
        n_qubits=config.model.n_rules + data_bundle.metadata["n_classes"],
    )
    model = FuzzyQuantumClassifier(
        d_in=data_bundle.metadata["d_in"],
        n_fuzzy_sets=config.model.n_fuzzy_sets,
        n_rules=config.model.n_rules,
        n_classes=data_bundle.metadata["n_classes"],
        n_quantum_layers=config.model.n_quantum_layers,
        seed=config.seed,
        dev_name=runtime.quantum_device,
        diff_method=runtime.diff_method,
    ).to(runtime.torch_device)

    result = train_model(
        model=model,
        train_loader=data_bundle.train_loader,
        val_loader=data_bundle.val_loader,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        lambda_sigma=config.training.lambda_sigma,
        torch_device=runtime.torch_device,
        run_dir=temp_dir,
        show_progress=False,
        early_stopping=config.training.early_stopping,
        patience=config.training.patience,
        min_delta=config.training.min_delta,
    )

    assert result.best_epoch == 1
    assert result.best_checkpoint_path.exists()
    assert result.last_checkpoint_path.exists()
    assert result.history.epoch == [1]


def test_training_execution_respects_output_flags(temp_dir: Path) -> None:
    config = _build_smoke_config(temp_dir)
    config.logging.save_history_csv = False
    config.logging.save_history_json = False
    config.logging.plot_loss = False
    config.logging.plot_accuracy = False
    config.outputs.save_best_model = False
    config.outputs.save_last_model = False
    config.outputs.save_rules = False
    config.outputs.save_fuzzy_parameters = False

    execution = run_training_experiment(config)

    assert not (execution.run_dir / "history.csv").exists()
    assert not (execution.run_dir / "history.json").exists()
    assert not (execution.run_dir / "loss_curve.png").exists()
    assert not (execution.run_dir / "accuracy_curve.png").exists()
    assert not (execution.run_dir / "best_model.pt").exists()
    assert not (execution.run_dir / "model.pt").exists()
    assert not (execution.run_dir / "rules.json").exists()
    assert not (execution.run_dir / "fuzzy_parameters.json").exists()

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .artifacts import (
    apply_output_retention_policy,
    create_run_dir,
    plot_accuracy_curves,
    plot_loss_curves,
    print_experiment_summary,
    print_final_metrics,
    save_history,
)
from .baselines import run_baselines
from .config import (
    ConfigurationError,
    ExperimentConfig,
    apply_cli_overrides,
    load_experiment_config,
    save_experiment_config,
    validate_experiment_config,
)
from .data import build_dataloaders, make_noisy_dataloader
from .device import collect_device_info, resolve_runtime, run_runtime_diagnostics
from .evaluate import evaluate_test_set, inspect_sample
from .model import FuzzyQuantumClassifier
from .train import TrainResult, load_checkpoint, train_model
from .utils import configure_torch_determinism, save_json, set_seed, timestamp_string


@dataclass(slots=True)
class TrainingExecution:
    config: ExperimentConfig
    run_dir: Path
    train_result: TrainResult
    test_metrics: dict[str, Any]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fuzzy QNN project CLI.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    _build_check_gpu_parser(
        subparsers.add_parser("check-gpu", help="Validate Torch/PennyLane runtime.")
    )
    _build_train_parser(subparsers.add_parser("train", help="Train the FQNN model."))
    _build_benchmark_parser(subparsers.add_parser("benchmark", help="Benchmark FQNN training."))
    _build_evaluate_parser(subparsers.add_parser("evaluate", help="Evaluate a saved run."))
    _build_evaluate_noise_parser(
        subparsers.add_parser(
            "evaluate-noise",
            help="Evaluate robustness to noise.",
        )
    )
    _build_run_experiments_parser(
        subparsers.add_parser("run-experiments", help="Run multiple configs.")
    )
    _build_inspect_sample_parser(
        subparsers.add_parser("inspect-sample", help="Inspect a saved model sample.")
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    command = str(args.command)
    if command == "check-gpu":
        return _run_check_gpu(args)
    if command == "train":
        return _run_train(args)
    if command == "benchmark":
        return _run_benchmark(args)
    if command == "evaluate":
        return _run_evaluate(args)
    if command == "evaluate-noise":
        return _run_evaluate_noise(args)
    if command == "run-experiments":
        return _run_experiments(args)
    if command == "inspect-sample":
        return _run_inspect_sample(args)
    parser.error(f"Unknown command {command}")
    return 1


def check_gpu_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate GPU runtime for FQNN.")
    _build_check_gpu_parser(parser)
    return _run_check_gpu(parser.parse_args(list(argv) if argv is not None else None))


def train_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an FQNN experiment from YAML config.")
    _build_train_parser(parser)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return _run_train(args)
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2
    except RuntimeError as error:
        print(_format_runtime_error(error, args), file=sys.stderr)
        return 1


def benchmark_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark FQNN training.")
    _build_benchmark_parser(parser)
    return _run_benchmark(parser.parse_args(list(argv) if argv is not None else None))


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved FQNN run.")
    _build_evaluate_parser(parser)
    return _run_evaluate(parser.parse_args(list(argv) if argv is not None else None))


def evaluate_noise_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate robustness under input noise.")
    _build_evaluate_noise_parser(parser)
    return _run_evaluate_noise(parser.parse_args(list(argv) if argv is not None else None))


def run_experiments_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run multiple FQNN experiments.")
    _build_run_experiments_parser(parser)
    return _run_experiments(parser.parse_args(list(argv) if argv is not None else None))


def inspect_sample_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a saved FQNN sample.")
    _build_inspect_sample_parser(parser)
    return _run_inspect_sample(parser.parse_args(list(argv) if argv is not None else None))


def _run_check_gpu(args: argparse.Namespace) -> int:
    diagnostic = run_runtime_diagnostics(require_gpu=bool(args.require_gpu))
    for check in diagnostic.checks:
        prefix = "[OK]" if check.ok else "[ERROR]"
        print(f"{prefix} {check.name}: {check.detail}")
    return 0 if diagnostic.ok else 1


def _run_train(args: argparse.Namespace) -> int:
    config = _load_and_override_config(args)
    if getattr(args, "dry_run", False):
        dry_run_report = run_training_dry_run(config)
        print_dry_run_report(dry_run_report)
        return 0
    execution = run_training_experiment(config)
    print_final_metrics(execution.test_metrics, execution.run_dir)
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    config = _load_and_override_config(args)
    execution = run_training_experiment(config)
    metrics = execution.test_metrics
    peak_cuda_memory_mb = (
        float(torch.cuda.max_memory_allocated() / (1024**2))
        if torch.cuda.is_available() and metrics["torch_device"] == "cuda"
        else 0.0
    )
    print("Benchmark summary")
    print("-----------------")
    print(f"seconds_per_epoch_mean: {metrics['seconds_per_epoch_mean']:.4f}")
    print(f"seconds_per_epoch_std:  {metrics['seconds_per_epoch_std']:.4f}")
    samples_per_second = execution.config.training.batch_size / max(
        metrics["seconds_per_epoch_mean"],
        1e-8,
    )
    print(f"samples_per_second:     {samples_per_second:.4f}")
    print(f"n_qubits:               {metrics['n_qubits']}")
    print(f"batch_size:             {execution.config.training.batch_size}")
    print(f"quantum_device:         {metrics['quantum_device']}")
    print(f"torch_device:           {metrics['torch_device']}")
    print(f"peak_cuda_memory_mb:    {peak_cuda_memory_mb:.4f}")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    config = load_experiment_config(run_dir / "config.yaml")
    rebuilt = _rebuild_run(config=config, run_dir=run_dir)
    metrics = rebuilt["metrics"]
    if args.overwrite or not (run_dir / "test_metrics.json").exists():
        save_json(run_dir / "test_metrics.json", metrics)
        save_json(run_dir / "metrics.json", metrics)
    print_final_metrics(metrics, run_dir)
    return 0


def _run_evaluate_noise(args: argparse.Namespace) -> int:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        config = load_experiment_config(run_dir / "config.yaml")
        rebuilt = _rebuild_run(config=config, run_dir=run_dir)
        bundle = rebuilt["data_bundle"]
        runtime = rebuilt["runtime"]
        model = rebuilt["model"]
        output_run_dir = run_dir
    else:
        config = _load_and_override_config(args)
        execution = run_training_experiment(config)
        rebuilt = _rebuild_run(config=config, run_dir=execution.run_dir)
        bundle = rebuilt["data_bundle"]
        runtime = rebuilt["runtime"]
        model = rebuilt["model"]
        output_run_dir = execution.run_dir

    noise_results: list[dict[str, Any]] = []
    for index, noise_std in enumerate(args.noise_std):
        noisy_loader = make_noisy_dataloader(
            dataset=bundle.test_dataset,
            batch_size=config.training.batch_size,
            noise_std=float(noise_std),
            seed=config.seed + index,
        )
        metrics = evaluate_test_set(
            model=model,
            test_loader=noisy_loader,
            torch_device=runtime.torch_device,
            class_names=bundle.metadata.get("class_names"),
            n_qubits=runtime.n_qubits,
            n_rules=config.model.n_rules,
            n_quantum_layers=config.model.n_quantum_layers,
            quantum_device=runtime.quantum_device,
            diff_method=runtime.diff_method,
        ).to_dict()
        metrics["noise_std"] = float(noise_std)
        noise_results.append(metrics)
        print(
            f"noise_std={noise_std:.4f} "
            f"accuracy={metrics['test_accuracy']:.4f} "
            f"loss={metrics['test_loss']:.4f}"
        )

    save_json(output_run_dir / "noise_metrics.json", noise_results)
    return 0


def _run_experiments(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for config_path in args.configs:
        config = load_experiment_config(config_path)
        execution = run_training_experiment(config)
        rows.append(
            {
                "experiment_name": config.experiment_name,
                "dataset": config.dataset.name,
                "model_type": "fuzzy_quantum",
                "seed": config.seed,
                "accuracy": execution.test_metrics["test_accuracy"],
                "loss": execution.test_metrics["test_loss"],
                "n_qubits": execution.test_metrics["n_qubits"],
                "n_rules": execution.test_metrics["n_rules"],
                "n_quantum_layers": execution.test_metrics["n_quantum_layers"],
                "quantum_device": execution.test_metrics["quantum_device"],
                "torch_device": execution.test_metrics["torch_device"],
                "train_seconds": execution.test_metrics["train_seconds"],
            }
        )

        rebuilt = _rebuild_run(config=config, run_dir=execution.run_dir)
        for baseline_metrics in run_baselines(rebuilt["data_bundle"], config):
            rows.append(
                {
                    "experiment_name": config.experiment_name,
                    "dataset": config.dataset.name,
                    "model_type": baseline_metrics["model_type"],
                    "seed": config.seed,
                    "accuracy": baseline_metrics["accuracy"],
                    "loss": baseline_metrics["loss"],
                    "n_qubits": execution.test_metrics["n_qubits"],
                    "n_rules": config.model.n_rules,
                    "n_quantum_layers": config.model.n_quantum_layers,
                    "quantum_device": execution.test_metrics["quantum_device"],
                    "torch_device": execution.test_metrics["torch_device"],
                    "train_seconds": execution.test_metrics["train_seconds"],
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "summary.csv", index=False)
    return 0


def _run_inspect_sample(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    config = load_experiment_config(run_dir / "config.yaml")
    rebuilt = _rebuild_run(config=config, run_dir=run_dir)
    bundle = rebuilt["data_bundle"]
    model = rebuilt["model"]
    sample_tensor = (
        bundle.test_dataset.tensors[0][int(args.sample_index)]
        .unsqueeze(0)
        .to(rebuilt["runtime"].torch_device)
    )
    inspection = inspect_sample(
        model=model,
        x=sample_tensor,
        class_names=bundle.metadata.get("class_names"),
    )
    print(inspection)
    return 0


def run_training_experiment(config: ExperimentConfig) -> TrainingExecution:
    set_seed(config.seed)
    configure_torch_determinism(config.training.deterministic)
    run_dir = create_run_dir(
        experiment_name=config.experiment_name,
        output_dir=config.outputs.output_dir,
        timestamp=timestamp_string(),
    )
    save_experiment_config(config, run_dir / "config.yaml")
    data_bundle = build_dataloaders(config)
    runtime = resolve_runtime(
        config.runtime,
        config.model.n_rules + data_bundle.metadata["n_classes"],
    )
    runtime_info = collect_device_info(runtime)
    runtime_info["n_qubits"] = runtime.n_qubits
    save_json(run_dir / "device_info.json", runtime_info)
    print_experiment_summary(config, data_bundle.metadata, runtime_info, run_dir)

    model = FuzzyQuantumClassifier(
        d_in=data_bundle.metadata["d_in"],
        n_fuzzy_sets=config.model.n_fuzzy_sets,
        n_rules=config.model.n_rules,
        n_classes=data_bundle.metadata["n_classes"],
        n_quantum_layers=config.model.n_quantum_layers,
        seed=config.seed,
        dev_name=runtime.quantum_device,
        diff_method=runtime.diff_method,
        quantum_inputs_on_cpu=runtime.quantum_inputs_on_cpu,
    ).to(runtime.torch_device)

    train_result = train_model(
        model=model,
        train_loader=data_bundle.train_loader,
        val_loader=data_bundle.val_loader,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        lambda_sigma=config.training.lambda_sigma,
        torch_device=runtime.torch_device,
        run_dir=run_dir,
        show_progress=config.logging.show_progress,
        early_stopping=config.training.early_stopping,
        patience=config.training.patience,
        min_delta=config.training.min_delta,
    )

    if config.logging.save_history_csv or config.logging.save_history_json:
        save_history(
            train_result.history,
            run_dir,
            save_csv=config.logging.save_history_csv,
            save_json_copy=config.logging.save_history_json,
        )
    if config.logging.plot_loss:
        plot_loss_curves(train_result.history, run_dir / "loss_curve.png")
    if config.logging.plot_accuracy:
        plot_accuracy_curves(train_result.history, run_dir / "accuracy_curve.png")
    if config.outputs.save_rules:
        save_json(run_dir / "rules.json", model.fuzzy_block.rule_layer.export_rules())
    if config.outputs.save_fuzzy_parameters:
        save_json(run_dir / "fuzzy_parameters.json", model.export_fuzzy_parameters())

    load_checkpoint(model, train_result.best_checkpoint_path, runtime.torch_device)
    test_metrics = evaluate_test_set(
        model=model,
        test_loader=data_bundle.test_loader,
        torch_device=runtime.torch_device,
        class_names=data_bundle.metadata.get("class_names"),
        train_seconds=train_result.train_seconds,
        seconds_per_epoch_mean=train_result.seconds_per_epoch_mean,
        seconds_per_epoch_std=train_result.seconds_per_epoch_std,
        n_qubits=runtime.n_qubits,
        n_rules=config.model.n_rules,
        n_quantum_layers=config.model.n_quantum_layers,
        quantum_device=runtime.quantum_device,
        diff_method=runtime.diff_method,
    ).to_dict()
    save_json(run_dir / "test_metrics.json", test_metrics)
    save_json(run_dir / "metrics.json", test_metrics)
    apply_output_retention_policy(
        train_result,
        save_best_model=config.outputs.save_best_model,
        save_last_model=config.outputs.save_last_model,
    )
    return TrainingExecution(
        config=config,
        run_dir=run_dir,
        train_result=train_result,
        test_metrics=test_metrics,
    )


def run_training_dry_run(config: ExperimentConfig) -> dict[str, Any]:
    validate_experiment_config(config)
    data_bundle = build_dataloaders(config)
    runtime = resolve_runtime(
        config.runtime,
        config.model.n_rules + data_bundle.metadata["n_classes"],
    )
    return {
        "experiment_name": config.experiment_name,
        "dataset": config.dataset.name,
        "torch_device": runtime.torch_device.type,
        "quantum_device": runtime.quantum_device,
        "diff_method": runtime.diff_method,
        "n_qubits": runtime.n_qubits,
        "n_rules": config.model.n_rules,
        "n_classes": data_bundle.metadata["n_classes"],
        "output_dir": str(Path(config.outputs.output_dir) / config.experiment_name),
        "n_train": data_bundle.metadata["n_train"],
        "n_val": data_bundle.metadata["n_val"],
        "n_test": data_bundle.metadata["n_test"],
    }


def _load_and_override_config(args: argparse.Namespace) -> ExperimentConfig:
    config = load_experiment_config(args.config)
    updated_config = apply_cli_overrides(config, args)
    validate_experiment_config(updated_config, config_path=args.config)
    return updated_config


def print_dry_run_report(report: dict[str, Any]) -> None:
    print("Dry run successful")
    print("------------------")
    print(f"experiment_name: {report['experiment_name']}")
    print(f"dataset:         {report['dataset']}")
    print(f"n_train:         {report['n_train']}")
    print(f"n_val:           {report['n_val']}")
    print(f"n_test:          {report['n_test']}")
    print(f"n_classes:       {report['n_classes']}")
    print(f"n_rules:         {report['n_rules']}")
    print(f"n_qubits:        {report['n_qubits']}")
    print(f"torch_device:    {report['torch_device']}")
    print(f"quantum_device:  {report['quantum_device']}")
    print(f"diff_method:     {report['diff_method']}")
    print(f"output_dir:      {report['output_dir']}")
    print("No training was executed.")


def _rebuild_run(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    set_seed(config.seed)
    configure_torch_determinism(config.training.deterministic)
    data_bundle = build_dataloaders(config)
    runtime = resolve_runtime(
        config.runtime,
        config.model.n_rules + data_bundle.metadata["n_classes"],
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
        quantum_inputs_on_cpu=runtime.quantum_inputs_on_cpu,
    ).to(runtime.torch_device)
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "model.pt"
    load_checkpoint(model, checkpoint_path, runtime.torch_device)
    metrics = evaluate_test_set(
        model=model,
        test_loader=data_bundle.test_loader,
        torch_device=runtime.torch_device,
        class_names=data_bundle.metadata.get("class_names"),
        n_qubits=runtime.n_qubits,
        n_rules=config.model.n_rules,
        n_quantum_layers=config.model.n_quantum_layers,
        quantum_device=runtime.quantum_device,
        diff_method=runtime.diff_method,
    ).to_dict()
    return {
        "data_bundle": data_bundle,
        "runtime": runtime,
        "model": model,
        "metrics": metrics,
    }


def _build_check_gpu_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--require-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if a GPU runtime is not available.",
    )


def _build_train_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, dataset, and devices without training.",
    )
    _add_common_overrides(parser)


def _build_benchmark_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    _add_common_overrides(parser)


def _build_evaluate_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True, help="Path to a saved run directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite test_metrics.json.")


def _build_evaluate_noise_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", help="Path to a saved run directory.")
    parser.add_argument("--config", help="Train from this config before evaluating noise.")
    parser.add_argument(
        "--noise-std",
        nargs="*",
        type=float,
        default=[0.01, 0.05, 0.10],
        help="Noise standard deviations applied to normalized inputs.",
    )
    _add_common_overrides(parser)


def _build_run_experiments_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--configs", nargs="+", required=True, help="List of config files to run.")
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Directory where summary.csv is saved.",
    )


def _build_inspect_sample_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True, help="Path to a saved run directory.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index inside the test set.",
    )


def _add_common_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-epochs", type=int, dest="max_epochs")
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--learning-rate", type=float, dest="learning_rate")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--experiment-name")
    parser.add_argument("--n-samples", type=int, dest="n_samples")
    parser.add_argument("--class-sep", type=float, dest="class_sep")
    parser.add_argument("--flip-y", type=float, dest="flip_y")
    parser.add_argument("--torch-device")
    parser.add_argument("--quantum-device")
    parser.add_argument("--diff-method")
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefer-gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plot-loss", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=None)


def _format_runtime_error(error: RuntimeError, args: argparse.Namespace) -> str:
    message = str(error)
    requested_quantum_device = getattr(args, "quantum_device", None)
    require_gpu = getattr(args, "require_gpu", None)
    if requested_quantum_device == "lightning.gpu" or require_gpu:
        return (
            "Runtime error while preparing the quantum backend.\n"
            f"{message}\n"
            "Suggestion: verify that pennylane-lightning-gpu is installed and that "
            "CUDA-enabled PyTorch can see your GPU. You can also retry with "
            "--no-require-gpu or --quantum-device lightning.qubit for CPU fallback."
        )
    return f"Runtime error:\n{message}"

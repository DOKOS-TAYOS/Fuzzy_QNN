from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    pass


@dataclass(slots=True)
class DatasetConfig:
    name: str
    test_size: float
    val_size: float
    scale: str
    feature_reduction: str
    d_in: int
    n_samples: int | None = None
    n_classes: int | None = None
    n_informative: int | None = None
    n_redundant: int | None = None
    class_sep: float | None = None
    flip_y: float | None = None


@dataclass(slots=True)
class ModelConfig:
    n_fuzzy_sets: int
    n_rules: int
    n_quantum_layers: int


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    lambda_sigma: float
    early_stopping: bool
    patience: int
    min_delta: float
    deterministic: bool = False


@dataclass(slots=True)
class RuntimeConfig:
    prefer_gpu: bool
    require_gpu: bool
    torch_device: str
    quantum_device: str
    diff_method: str
    quantum_inputs_on_cpu: bool


@dataclass(slots=True)
class LoggingConfig:
    show_progress: bool
    plot_loss: bool
    plot_accuracy: bool
    save_history_csv: bool
    save_history_json: bool
    save_stdout_log: bool


@dataclass(slots=True)
class OutputsConfig:
    output_dir: str
    save_best_model: bool
    save_last_model: bool
    save_rules: bool
    save_fuzzy_parameters: bool


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    runtime: RuntimeConfig
    logging: LoggingConfig
    outputs: OutputsConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"Configuration error in '{config_path}': invalid YAML syntax. {error}"
        ) from error
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Configuration error in '{config_path}': expected a top-level mapping."
        )

    dataset_raw = _mapping(raw.get("dataset"), "dataset", config_path)
    model_raw = _mapping(raw.get("model"), "model", config_path)
    training_raw = _mapping(raw.get("training"), "training", config_path)
    runtime_raw = _mapping(raw.get("runtime"), "runtime", config_path)
    logging_raw = _mapping(raw.get("logging") or {}, "logging", config_path)
    outputs_raw = _mapping(raw.get("outputs") or {}, "outputs", config_path)

    config = ExperimentConfig(
        experiment_name=str(_required_value(raw, "experiment_name", "root", config_path)),
        seed=int(_required_value(raw, "seed", "root", config_path)),
        dataset=DatasetConfig(
            name=str(_required_value(dataset_raw, "name", "dataset", config_path)),
            test_size=float(_required_value(dataset_raw, "test_size", "dataset", config_path)),
            val_size=float(_required_value(dataset_raw, "val_size", "dataset", config_path)),
            scale=str(_required_value(dataset_raw, "scale", "dataset", config_path)),
            feature_reduction=str(
                _required_value(dataset_raw, "feature_reduction", "dataset", config_path)
            ),
            d_in=int(_required_value(dataset_raw, "d_in", "dataset", config_path)),
            n_samples=_optional_int(dataset_raw.get("n_samples")),
            n_classes=_optional_int(dataset_raw.get("n_classes")),
            n_informative=_optional_int(dataset_raw.get("n_informative")),
            n_redundant=_optional_int(dataset_raw.get("n_redundant")),
            class_sep=_optional_float(dataset_raw.get("class_sep")),
            flip_y=_optional_float(dataset_raw.get("flip_y")),
        ),
        model=ModelConfig(
            n_fuzzy_sets=int(_required_value(model_raw, "n_fuzzy_sets", "model", config_path)),
            n_rules=int(_required_value(model_raw, "n_rules", "model", config_path)),
            n_quantum_layers=int(
                _required_value(model_raw, "n_quantum_layers", "model", config_path)
            ),
        ),
        training=TrainingConfig(
            batch_size=int(_required_value(training_raw, "batch_size", "training", config_path)),
            epochs=int(_required_value(training_raw, "epochs", "training", config_path)),
            learning_rate=float(
                _required_value(training_raw, "learning_rate", "training", config_path)
            ),
            weight_decay=float(training_raw.get("weight_decay", 0.0)),
            lambda_sigma=float(training_raw.get("lambda_sigma", 0.0)),
            early_stopping=bool(training_raw.get("early_stopping", True)),
            patience=int(training_raw.get("patience", 20)),
            min_delta=float(training_raw.get("min_delta", 1e-5)),
            deterministic=bool(training_raw.get("deterministic", False)),
        ),
        runtime=RuntimeConfig(
            prefer_gpu=bool(runtime_raw.get("prefer_gpu", False)),
            require_gpu=bool(runtime_raw.get("require_gpu", False)),
            torch_device=str(runtime_raw.get("torch_device", "cpu")),
            quantum_device=str(runtime_raw.get("quantum_device", "lightning.qubit")),
            diff_method=str(runtime_raw.get("diff_method", "adjoint")),
            quantum_inputs_on_cpu=bool(runtime_raw.get("quantum_inputs_on_cpu", False)),
        ),
        logging=LoggingConfig(
            show_progress=bool(logging_raw.get("show_progress", True)),
            plot_loss=bool(logging_raw.get("plot_loss", True)),
            plot_accuracy=bool(logging_raw.get("plot_accuracy", True)),
            save_history_csv=bool(logging_raw.get("save_history_csv", True)),
            save_history_json=bool(logging_raw.get("save_history_json", True)),
            save_stdout_log=bool(logging_raw.get("save_stdout_log", False)),
        ),
        outputs=OutputsConfig(
            output_dir=str(outputs_raw.get("output_dir", "runs")),
            save_best_model=bool(outputs_raw.get("save_best_model", True)),
            save_last_model=bool(outputs_raw.get("save_last_model", True)),
            save_rules=bool(outputs_raw.get("save_rules", True)),
            save_fuzzy_parameters=bool(outputs_raw.get("save_fuzzy_parameters", True)),
        ),
    )
    validate_experiment_config(config, config_path=config_path)
    return config


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    config_path = Path(path)
    config_path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def apply_cli_overrides(config: ExperimentConfig, args: Namespace) -> ExperimentConfig:
    updated_config = config

    if getattr(args, "experiment_name", None):
        updated_config = replace(updated_config, experiment_name=str(args.experiment_name))
    if getattr(args, "seed", None) is not None:
        updated_config = replace(updated_config, seed=int(args.seed))

    dataset = updated_config.dataset
    if getattr(args, "n_samples", None) is not None:
        dataset = replace(dataset, n_samples=int(args.n_samples))
    if getattr(args, "class_sep", None) is not None:
        dataset = replace(dataset, class_sep=float(args.class_sep))
    if getattr(args, "flip_y", None) is not None:
        dataset = replace(dataset, flip_y=float(args.flip_y))
    if dataset is not updated_config.dataset:
        updated_config = replace(updated_config, dataset=dataset)

    training = updated_config.training
    if getattr(args, "epochs", None) is not None:
        training = replace(training, epochs=int(args.epochs))
    if getattr(args, "max_epochs", None) is not None:
        training = replace(training, epochs=int(args.max_epochs))
    if getattr(args, "batch_size", None) is not None:
        training = replace(training, batch_size=int(args.batch_size))
    if getattr(args, "learning_rate", None) is not None:
        training = replace(training, learning_rate=float(args.learning_rate))
    if training is not updated_config.training:
        updated_config = replace(updated_config, training=training)

    runtime = updated_config.runtime
    if getattr(args, "require_gpu", None) is not None:
        runtime = replace(runtime, require_gpu=bool(args.require_gpu))
    if getattr(args, "prefer_gpu", None) is not None:
        runtime = replace(runtime, prefer_gpu=bool(args.prefer_gpu))
    if getattr(args, "torch_device", None):
        runtime = replace(runtime, torch_device=str(args.torch_device))
    if getattr(args, "quantum_device", None):
        runtime = replace(runtime, quantum_device=str(args.quantum_device))
    if getattr(args, "diff_method", None):
        runtime = replace(runtime, diff_method=str(args.diff_method))
    if runtime is not updated_config.runtime:
        updated_config = replace(updated_config, runtime=runtime)

    logging = updated_config.logging
    if getattr(args, "plot_loss", None) is not None:
        logging = replace(logging, plot_loss=bool(args.plot_loss))
    if getattr(args, "show_progress", None) is not None:
        logging = replace(logging, show_progress=bool(args.show_progress))
    if logging is not updated_config.logging:
        updated_config = replace(updated_config, logging=logging)

    outputs = updated_config.outputs
    if getattr(args, "output_dir", None):
        outputs = replace(outputs, output_dir=str(args.output_dir))
    if outputs is not updated_config.outputs:
        updated_config = replace(updated_config, outputs=outputs)

    return updated_config


def validate_experiment_config(
    config: ExperimentConfig,
    *,
    config_path: str | Path | None = None,
) -> None:
    prefix = "Configuration error"
    if config_path is not None:
        prefix = f"{prefix} in '{config_path}'"

    problems: list[str] = []
    if config.dataset.name not in {"iris", "breast_cancer", "wine", "synthetic"}:
        problems.append("dataset.name must be one of: iris, breast_cancer, wine, synthetic.")
    if config.dataset.scale not in {"minmax", "standard_then_minmax"}:
        problems.append("dataset.scale must be 'minmax' or 'standard_then_minmax'.")
    if config.dataset.feature_reduction not in {"none", "pca"}:
        problems.append("dataset.feature_reduction must be 'none' or 'pca'.")
    if config.dataset.d_in <= 0:
        problems.append("dataset.d_in must be a positive integer.")
    if not 0.0 < config.dataset.test_size < 1.0:
        problems.append("dataset.test_size must be between 0 and 1.")
    if not 0.0 <= config.dataset.val_size < 1.0:
        problems.append("dataset.val_size must be between 0 and 1.")
    if (config.dataset.test_size + config.dataset.val_size) >= 1.0:
        problems.append("dataset.test_size + dataset.val_size must be smaller than 1.")
    if config.model.n_fuzzy_sets <= 0:
        problems.append("model.n_fuzzy_sets must be a positive integer.")
    if config.model.n_rules <= 0:
        problems.append("model.n_rules must be a positive integer.")
    if config.model.n_quantum_layers <= 0:
        problems.append("model.n_quantum_layers must be a positive integer.")
    if config.training.batch_size <= 0:
        problems.append("training.batch_size must be a positive integer.")
    if config.training.epochs <= 0:
        problems.append("training.epochs must be a positive integer.")
    if config.training.learning_rate <= 0.0:
        problems.append("training.learning_rate must be greater than 0.")
    if config.training.patience < 0:
        problems.append("training.patience must be 0 or greater.")
    if config.training.min_delta < 0.0:
        problems.append("training.min_delta must be 0 or greater.")
    if config.runtime.diff_method not in {"adjoint", "backprop"}:
        problems.append("runtime.diff_method must be 'adjoint' or 'backprop'.")
    if not config.outputs.output_dir:
        problems.append("outputs.output_dir cannot be empty.")

    if problems:
        joined_problems = "\n".join(f"- {problem}" for problem in problems)
        raise ConfigurationError(f"{prefix}:\n{joined_problems}")


def _mapping(value: Any, name: str, config_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(
            f"Configuration error in '{config_path}': section '{name}' must be a mapping."
        )
    return value


def _required_value(
    mapping: dict[str, Any],
    key: str,
    section_name: str,
    config_path: Path,
) -> Any:
    if key not in mapping:
        section_label = "top level" if section_name == "root" else f"section '{section_name}'"
        raise ConfigurationError(
            f"Configuration error in '{config_path}': missing key '{key}' in {section_label}."
        )
    return mapping[key]


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)

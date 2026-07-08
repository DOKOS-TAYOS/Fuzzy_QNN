from __future__ import annotations

from pathlib import Path

import pytest

from fuzzy_qnn.config import ConfigurationError, load_experiment_config


def test_load_experiment_config_reports_missing_top_level_key(tmp_path: Path) -> None:
    config_path = tmp_path / "missing_name.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 1234",
                "dataset:",
                "  name: synthetic",
                "  test_size: 0.2",
                "  val_size: 0.2",
                "  scale: minmax",
                "  feature_reduction: none",
                "  d_in: 4",
                "model:",
                "  n_fuzzy_sets: 3",
                "  n_rules: 4",
                "  n_quantum_layers: 1",
                "training:",
                "  batch_size: 8",
                "  epochs: 1",
                "  learning_rate: 0.01",
                "runtime:",
                "  prefer_gpu: false",
                "  require_gpu: false",
                "  torch_device: cpu",
                "  quantum_device: default.qubit",
                "  diff_method: backprop",
                "outputs:",
                "  output_dir: runs",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="missing key 'experiment_name'"):
        load_experiment_config(config_path)


def test_load_experiment_config_reports_missing_nested_key(tmp_path: Path) -> None:
    config_path = tmp_path / "missing_dataset_field.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: test",
                "seed: 1234",
                "dataset:",
                "  name: synthetic",
                "  test_size: 0.2",
                "  val_size: 0.2",
                "  scale: minmax",
                "model:",
                "  n_fuzzy_sets: 3",
                "  n_rules: 4",
                "  n_quantum_layers: 1",
                "training:",
                "  batch_size: 8",
                "  epochs: 1",
                "  learning_rate: 0.01",
                "runtime:",
                "  prefer_gpu: false",
                "  require_gpu: false",
                "  torch_device: cpu",
                "  quantum_device: default.qubit",
                "  diff_method: backprop",
                "outputs:",
                "  output_dir: runs",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="missing key 'feature_reduction' in section 'dataset'",
    ):
        load_experiment_config(config_path)

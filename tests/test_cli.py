from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _write_smoke_config(config_path: Path, output_dir: Path) -> None:
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: smoke_cli",
                "seed: 1234",
                "",
                "dataset:",
                "  name: synthetic",
                "  test_size: 0.20",
                "  val_size: 0.20",
                "  scale: minmax",
                "  feature_reduction: none",
                "  d_in: 4",
                "  n_samples: 48",
                "  n_classes: 2",
                "  n_informative: 3",
                "  n_redundant: 1",
                "  class_sep: 1.0",
                "  flip_y: 0.01",
                "",
                "model:",
                "  n_fuzzy_sets: 3",
                "  n_rules: 4",
                "  n_quantum_layers: 1",
                "",
                "training:",
                "  batch_size: 8",
                "  epochs: 1",
                "  learning_rate: 0.01",
                "  weight_decay: 0.0",
                "  lambda_sigma: 0.0",
                "  early_stopping: false",
                "  patience: 3",
                "  min_delta: 0.00001",
                "",
                "runtime:",
                "  prefer_gpu: false",
                "  require_gpu: false",
                "  torch_device: cpu",
                "  quantum_device: default.qubit",
                "  diff_method: backprop",
                "  quantum_inputs_on_cpu: false",
                "",
                "logging:",
                "  show_progress: false",
                "  plot_loss: true",
                "  plot_accuracy: false",
                "  save_history_csv: true",
                "  save_history_json: true",
                "  save_stdout_log: false",
                "",
                "outputs:",
                f"  output_dir: {output_dir.as_posix()}",
                "  save_best_model: true",
                "  save_last_model: true",
                "  save_rules: true",
                "  save_fuzzy_parameters: true",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_check_gpu_cli_runs_without_requiring_gpu(cli_env: dict[str, str]) -> None:
    completed_process = subprocess.run(
        [sys.executable, "scripts/check_gpu.py", "--no-require-gpu"],
        check=False,
        capture_output=True,
        text=True,
        env=cli_env,
    )

    assert completed_process.returncode == 0
    assert "Torch CUDA available" in completed_process.stdout


def test_train_cli_creates_basic_outputs(temp_dir: Path, cli_env: dict[str, str]) -> None:
    config_path = temp_dir / "smoke.yaml"
    output_dir = temp_dir / "runs"
    _write_smoke_config(config_path, output_dir)

    completed_process = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        env=cli_env,
    )

    assert completed_process.returncode == 0
    assert "Final test metrics" in completed_process.stdout
    run_root = output_dir / "smoke_cli"
    run_directories = list(run_root.iterdir())
    assert len(run_directories) == 1
    run_dir = run_directories[0]
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "test_metrics.json").exists()
    assert (run_dir / "best_model.pt").exists()
    assert (run_dir / "loss_curve.png").exists()


def test_train_cli_shows_progress_bar_in_terminal_output(
    temp_dir: Path,
    cli_env: dict[str, str],
) -> None:
    config_path = temp_dir / "smoke_progress.yaml"
    output_dir = temp_dir / "runs"
    _write_smoke_config(config_path, output_dir)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "show_progress: false", "show_progress: true"
        ),
        encoding="utf-8",
    )

    completed_process = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        env=cli_env,
    )

    assert completed_process.returncode == 0
    assert "Epoch 001/001" in completed_process.stdout


def test_train_cli_dry_run_validates_without_training(
    temp_dir: Path,
    cli_env: dict[str, str],
) -> None:
    config_path = temp_dir / "dry_run.yaml"
    output_dir = temp_dir / "runs"
    _write_smoke_config(config_path, output_dir)

    completed_process = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(config_path),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=cli_env,
    )

    assert completed_process.returncode == 0
    assert "Dry run successful" in completed_process.stdout
    assert "No training was executed" in completed_process.stdout
    run_root = output_dir / "smoke_cli"
    assert not run_root.exists()


def test_train_cli_max_epochs_and_experiment_name_override(
    temp_dir: Path,
    cli_env: dict[str, str],
) -> None:
    config_path = temp_dir / "override.yaml"
    output_dir = temp_dir / "runs"
    _write_smoke_config(config_path, output_dir)

    completed_process = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(config_path),
            "--max-epochs",
            "1",
            "--experiment-name",
            "cli_override",
            "--no-show-progress",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=cli_env,
    )

    assert completed_process.returncode == 0
    assert "name: cli_override" in completed_process.stdout
    run_root = output_dir / "cli_override"
    run_directories = list(run_root.iterdir())
    assert len(run_directories) == 1
    history_csv = (run_directories[0] / "history.csv").read_text(encoding="utf-8")
    assert history_csv.count("\n") == 2


def test_train_cli_reports_readable_config_error(temp_dir: Path, cli_env: dict[str, str]) -> None:
    bad_config_path = temp_dir / "bad.yaml"
    bad_config_path.write_text(
        "\n".join(
            [
                "experiment_name: broken",
                "seed: 1234",
                "dataset:",
                "  name: synthetic",
            ]
        ),
        encoding="utf-8",
    )

    completed_process = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(bad_config_path)],
        check=False,
        capture_output=True,
        text=True,
        env=cli_env,
    )

    assert completed_process.returncode != 0
    assert "Configuration error" in completed_process.stderr
    assert "training" in completed_process.stderr or "model" in completed_process.stderr

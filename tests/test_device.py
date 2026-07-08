from __future__ import annotations

import pytest
import torch

from fuzzy_qnn.config import RuntimeConfig
from fuzzy_qnn.device import resolve_runtime, run_runtime_diagnostics


def test_resolve_runtime_returns_cpu_when_gpu_not_preferred() -> None:
    runtime = resolve_runtime(
        RuntimeConfig(
            prefer_gpu=False,
            require_gpu=False,
            torch_device="cpu",
            quantum_device="lightning.qubit",
            diff_method="adjoint",
            quantum_inputs_on_cpu=False,
        ),
        n_qubits=6,
    )

    assert runtime.torch_device.type == "cpu"
    assert runtime.quantum_device in {"lightning.qubit", "default.qubit"}


def test_resolve_runtime_falls_back_cleanly_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    runtime = resolve_runtime(
        RuntimeConfig(
            prefer_gpu=True,
            require_gpu=False,
            torch_device="cuda",
            quantum_device="lightning.gpu",
            diff_method="adjoint",
            quantum_inputs_on_cpu=False,
        ),
        n_qubits=6,
    )

    assert runtime.torch_device.type == "cpu"
    assert runtime.quantum_device in {"lightning.qubit", "default.qubit"}


def test_resolve_runtime_raises_when_gpu_is_required_but_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError):
        resolve_runtime(
            RuntimeConfig(
                prefer_gpu=True,
                require_gpu=True,
                torch_device="cuda",
                quantum_device="lightning.gpu",
                diff_method="adjoint",
                quantum_inputs_on_cpu=False,
            ),
            n_qubits=6,
        )


def test_runtime_diagnostics_reports_minimal_qnode_forward_and_backward() -> None:
    diagnostics = run_runtime_diagnostics(require_gpu=False)
    check_names = {check.name for check in diagnostics.checks}

    assert "Minimal QNode forward" in check_names
    assert "Minimal QNode backward" in check_names


def test_runtime_diagnostics_explains_missing_required_gpu() -> None:
    diagnostics = run_runtime_diagnostics(require_gpu=True)

    assert diagnostics.ok is False
    details = "\n".join(check.detail for check in diagnostics.checks)
    assert "Suggestion" in details

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pennylane as qml
import torch

from .config import RuntimeConfig


@dataclass(slots=True)
class RuntimeInfo:
    torch_device: torch.device
    quantum_device: str
    diff_method: str
    n_qubits: int
    quantum_inputs_on_cpu: bool
    torch_cuda_available: bool
    torch_cuda_device_name: str | None


@dataclass(slots=True)
class DiagnosticCheck:
    name: str
    ok: bool
    detail: str


@dataclass(slots=True)
class DiagnosticResult:
    checks: list[DiagnosticCheck]
    runtime: RuntimeInfo | None

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def resolve_runtime(config: RuntimeConfig, n_qubits: int) -> RuntimeInfo:
    cuda_available = torch.cuda.is_available()
    wants_gpu = (
        config.prefer_gpu
        or config.torch_device == "cuda"
        or config.quantum_device == "lightning.gpu"
    )

    if config.require_gpu and not cuda_available:
        raise RuntimeError(
            "GPU is required but Torch CUDA is not available. "
            "Set require_gpu=false to allow CPU fallback."
        )

    torch_device = torch.device("cuda" if wants_gpu and cuda_available else "cpu")
    quantum_device = _resolve_quantum_device_name(
        requested_device=config.quantum_device,
        want_gpu=wants_gpu and cuda_available,
        require_gpu=config.require_gpu,
        n_qubits=n_qubits,
    )
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    return RuntimeInfo(
        torch_device=torch_device,
        quantum_device=quantum_device,
        diff_method=config.diff_method,
        n_qubits=n_qubits,
        quantum_inputs_on_cpu=config.quantum_inputs_on_cpu,
        torch_cuda_available=cuda_available,
        torch_cuda_device_name=device_name,
    )


def collect_device_info(runtime: RuntimeInfo) -> dict[str, Any]:
    total_memory_gb = None
    if runtime.torch_cuda_available:
        total_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 3)
    return {
        "torch_cuda_available": runtime.torch_cuda_available,
        "torch_cuda_device_name": runtime.torch_cuda_device_name,
        "torch_cuda_total_memory_gb": total_memory_gb,
        "torch_version": torch.__version__,
        "pennylane_version": qml.__version__,
        "torch_device": runtime.torch_device.type,
        "quantum_device": runtime.quantum_device,
        "diff_method": runtime.diff_method,
        "quantum_inputs_on_cpu": runtime.quantum_inputs_on_cpu,
    }


def run_runtime_diagnostics(require_gpu: bool = False) -> DiagnosticResult:
    checks: list[DiagnosticCheck] = []
    cuda_available = torch.cuda.is_available()
    checks.append(DiagnosticCheck("Torch CUDA available", True, str(cuda_available)))
    if cuda_available:
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        checks.append(
            DiagnosticCheck(
                "Torch CUDA device",
                True,
                str(torch.cuda.get_device_name(0)),
            )
        )
        checks.append(
            DiagnosticCheck(
                "Torch CUDA memory total",
                True,
                f"{total_memory_gb:.2f} GB",
            )
        )
    elif require_gpu:
        checks.append(
            DiagnosticCheck(
                "Torch CUDA device",
                False,
                "No CUDA device available but require_gpu=true. "
                "Suggestion: install CUDA-enabled PyTorch or use --no-require-gpu.",
            )
        )
        return DiagnosticResult(checks=checks, runtime=None)

    runtime_config = RuntimeConfig(
        prefer_gpu=require_gpu or cuda_available,
        require_gpu=require_gpu,
        torch_device="cuda" if require_gpu or cuda_available else "cpu",
        quantum_device="lightning.gpu" if require_gpu or cuda_available else "lightning.qubit",
        diff_method="adjoint",
        quantum_inputs_on_cpu=False,
    )

    try:
        runtime = resolve_runtime(runtime_config, n_qubits=6)
        checks.append(
            DiagnosticCheck(
                "PennyLane device creation",
                True,
                f"Using {runtime.quantum_device}",
            )
        )
    except Exception as error:
        detail = str(error)
        if require_gpu:
            detail = (
                f"{detail} Suggestion: install pennylane-lightning-gpu with a compatible "
                "CUDA stack, or run the CPU path with --no-require-gpu."
            )
        checks.append(DiagnosticCheck("PennyLane device creation", False, detail))
        return DiagnosticResult(checks=checks, runtime=None)

    try:
        _run_adjoint_check(runtime)
        checks.append(
            DiagnosticCheck("PennyLane adjoint differentiation", True, runtime.diff_method)
        )
    except Exception as error:
        checks.append(DiagnosticCheck("PennyLane adjoint differentiation", False, str(error)))
        return DiagnosticResult(checks=checks, runtime=runtime)

    try:
        _run_minimal_qnode_forward(runtime)
        checks.append(DiagnosticCheck("Minimal QNode forward", True, "OK"))
    except Exception as error:
        checks.append(DiagnosticCheck("Minimal QNode forward", False, str(error)))
        return DiagnosticResult(checks=checks, runtime=runtime)

    try:
        _run_minimal_qnode_backward(runtime)
        checks.append(DiagnosticCheck("Minimal QNode backward", True, "OK"))
    except Exception as error:
        checks.append(DiagnosticCheck("Minimal QNode backward", False, str(error)))
        return DiagnosticResult(checks=checks, runtime=runtime)

    try:
        _run_full_train_step(runtime)
        checks.append(DiagnosticCheck("Full FQNN one-batch train step", True, "OK"))
    except Exception as error:
        checks.append(DiagnosticCheck("Full FQNN one-batch train step", False, str(error)))
    return DiagnosticResult(checks=checks, runtime=runtime)


def _resolve_quantum_device_name(
    requested_device: str,
    want_gpu: bool,
    require_gpu: bool,
    n_qubits: int,
) -> str:
    if want_gpu:
        candidates = [requested_device, "lightning.gpu"]
        for candidate in dict.fromkeys(candidates):
            if _can_create_device(candidate, n_qubits):
                return candidate
        if require_gpu:
            raise RuntimeError(
                "No GPU-compatible PennyLane device could be initialized. "
                "Check pennylane-lightning-gpu and your CUDA stack."
            )
    for candidate in dict.fromkeys([requested_device, "lightning.qubit", "default.qubit"]):
        if _can_create_device(candidate, n_qubits):
            return candidate
    raise RuntimeError(
        "No compatible PennyLane device is available. "
        "Tried lightning.gpu/lightning.qubit/default.qubit."
    )


def _can_create_device(device_name: str, n_qubits: int) -> bool:
    try:
        qml.device(device_name, wires=n_qubits)
    except Exception:
        return False
    return True


def _run_adjoint_check(runtime: RuntimeInfo) -> None:
    device = qml.device(runtime.quantum_device, wires=2)

    @qml.qnode(device, interface="torch", diff_method=runtime.diff_method)
    def circuit(weights: torch.Tensor) -> Any:
        qml.RY(cast(Any, weights[0]), wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RZ(cast(Any, weights[1]), wires=1)
        return qml.expval(qml.PauliZ(1))

    weights = torch.randn(2, requires_grad=True, device=runtime.torch_device)
    _ = circuit(weights)


def _run_minimal_qnode_forward(runtime: RuntimeInfo) -> None:
    device = qml.device(runtime.quantum_device, wires=2)

    @qml.qnode(device, interface="torch", diff_method=runtime.diff_method)
    def circuit(weights: torch.Tensor) -> Any:
        qml.RY(cast(Any, weights[0]), wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RZ(cast(Any, weights[1]), wires=1)
        return qml.expval(qml.PauliZ(1))

    weights = torch.randn(2, device=runtime.torch_device)
    _ = circuit(weights)


def _run_minimal_qnode_backward(runtime: RuntimeInfo) -> None:
    device = qml.device(runtime.quantum_device, wires=2)

    @qml.qnode(device, interface="torch", diff_method=runtime.diff_method)
    def circuit(weights: torch.Tensor) -> Any:
        qml.RY(cast(Any, weights[0]), wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RZ(cast(Any, weights[1]), wires=1)
        return qml.expval(qml.PauliZ(1))

    weights = torch.randn(2, requires_grad=True, device=runtime.torch_device)
    result = cast(torch.Tensor, circuit(weights))
    result.backward()
    if weights.grad is None:
        raise RuntimeError("Torch gradient was not produced by the diagnostic QNode.")


def _run_full_train_step(runtime: RuntimeInfo) -> None:
    from .model import FuzzyQuantumClassifier

    model = FuzzyQuantumClassifier(
        d_in=4,
        n_fuzzy_sets=3,
        n_rules=4,
        n_classes=2,
        n_quantum_layers=1,
        seed=1234,
        dev_name=runtime.quantum_device,
        diff_method=runtime.diff_method,
    ).to(runtime.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    inputs = torch.rand(4, 4, device=runtime.torch_device)
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=runtime.torch_device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()

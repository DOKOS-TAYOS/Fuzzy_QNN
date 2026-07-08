from __future__ import annotations

import pennylane as qml
import pytest
import torch
import torch.nn.functional as functional

from fuzzy_qnn.model import FuzzyQuantumClassifier


def _has_lightning_gpu() -> bool:
    try:
        qml.device("lightning.gpu", wires=2)
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_lightning_gpu(),
    reason="CUDA o lightning.gpu no disponible",
)


def test_model_forward_and_backward_on_gpu() -> None:
    device = torch.device("cuda")
    model = FuzzyQuantumClassifier(
        d_in=4,
        n_fuzzy_sets=3,
        n_rules=4,
        n_classes=2,
        n_quantum_layers=1,
        seed=1234,
        dev_name="lightning.gpu",
        diff_method="adjoint",
    ).to(device)
    x = torch.rand(2, 4, device=device)
    y = torch.tensor([0, 1], dtype=torch.long, device=device)

    logits = model(x)
    loss = functional.cross_entropy(logits, y)
    loss.backward()

    assert model.theta.grad is not None
    assert model.phi.grad is not None

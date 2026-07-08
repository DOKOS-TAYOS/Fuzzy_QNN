from __future__ import annotations

import torch

from fuzzy_qnn.quantum import QuantumRuleCircuit


def test_quantum_circuit_forward_shape_and_range() -> None:
    circuit = QuantumRuleCircuit(
        n_rules=4,
        n_classes=2,
        n_layers=1,
        dev_name="default.qubit",
        diff_method="backprop",
    )
    alpha = torch.rand(3, 4)
    alpha = alpha / alpha.sum(dim=1, keepdim=True)
    theta = torch.randn(1, 4, 2, requires_grad=True)
    phi = torch.randn(1, 4, 2, requires_grad=True)

    outputs = circuit(alpha, theta, phi)

    assert outputs.shape == (3, 2)
    assert torch.all(outputs <= 1.0 + 1e-5)
    assert torch.all(outputs >= -1.0 - 1e-5)


def test_quantum_circuit_backward_pass() -> None:
    circuit = QuantumRuleCircuit(
        n_rules=3,
        n_classes=2,
        n_layers=1,
        dev_name="default.qubit",
        diff_method="backprop",
    )
    alpha = torch.rand(2, 3)
    alpha = alpha / alpha.sum(dim=1, keepdim=True)
    theta = torch.randn(1, 3, 2, requires_grad=True)
    phi = torch.randn(1, 3, 2, requires_grad=True)

    loss = circuit(alpha, theta, phi).sum()
    loss.backward()

    assert theta.grad is not None
    assert phi.grad is not None

from __future__ import annotations

import torch

from fuzzy_qnn.model import FuzzyQuantumClassifier


def test_model_forward_returns_logits_for_multiclass_problem() -> None:
    model = FuzzyQuantumClassifier(
        d_in=4,
        n_fuzzy_sets=3,
        n_rules=4,
        n_classes=3,
        n_quantum_layers=1,
        seed=1234,
        dev_name="default.qubit",
        diff_method="backprop",
    )
    x = torch.rand(3, 4)

    logits = model(x)

    assert logits.shape == (3, 3)


def test_model_can_return_intermediates() -> None:
    model = FuzzyQuantumClassifier(
        d_in=4,
        n_fuzzy_sets=3,
        n_rules=4,
        n_classes=2,
        n_quantum_layers=1,
        seed=1234,
        dev_name="default.qubit",
        diff_method="backprop",
    )
    x = torch.rand(2, 4)

    outputs = model(x, return_intermediates=True)

    assert set(outputs) == {"logits", "mu", "alpha", "z"}
    assert outputs["logits"].shape == (2, 2)
    assert outputs["mu"].shape == (2, 4, 3)
    assert outputs["alpha"].shape == (2, 4)
    assert outputs["z"].shape == (2, 2)


def test_model_accepts_quantum_inputs_on_cpu_flag() -> None:
    model = FuzzyQuantumClassifier(
        d_in=4,
        n_fuzzy_sets=3,
        n_rules=4,
        n_classes=2,
        n_quantum_layers=1,
        seed=1234,
        dev_name="default.qubit",
        diff_method="backprop",
        quantum_inputs_on_cpu=True,
    )
    x = torch.rand(2, 4)

    logits = model(x)

    assert logits.shape == (2, 2)
    assert logits.device == x.device
    assert model.quantum_circuit.quantum_inputs_on_cpu is True

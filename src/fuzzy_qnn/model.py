from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .fuzzy import FuzzyBlock
from .quantum import QuantumRuleCircuit


class FuzzyQuantumClassifier(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_fuzzy_sets: int,
        n_rules: int,
        n_classes: int,
        n_quantum_layers: int,
        seed: int = 1234,
        dev_name: str = "default.qubit",
        diff_method: str = "backprop",
        quantum_inputs_on_cpu: bool = False,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.n_quantum_layers = n_quantum_layers
        self.fuzzy_block = FuzzyBlock(
            d_in=d_in,
            n_fuzzy_sets=n_fuzzy_sets,
            n_rules=n_rules,
            seed=seed,
        )
        self.quantum_circuit = QuantumRuleCircuit(
            n_rules=n_rules,
            n_classes=n_classes,
            n_layers=n_quantum_layers,
            dev_name=dev_name,
            diff_method=diff_method,
            quantum_inputs_on_cpu=quantum_inputs_on_cpu,
        )
        self.theta = nn.Parameter(0.01 * torch.randn(n_quantum_layers, n_rules, 2))
        self.phi = nn.Parameter(0.01 * torch.randn(n_quantum_layers, n_rules, n_classes))
        self.gamma = nn.Parameter(torch.ones(n_classes))
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        memberships, activations = self.fuzzy_block(x)
        quantum_outputs = self.quantum_circuit(activations, self.theta, self.phi)
        logits = (self.gamma * quantum_outputs) + self.bias
        if not return_intermediates:
            return logits
        return {
            "logits": logits,
            "mu": memberships,
            "alpha": activations,
            "z": quantum_outputs,
        }

    def export_fuzzy_parameters(self) -> dict[str, Any]:
        return self.fuzzy_block.export_parameters()

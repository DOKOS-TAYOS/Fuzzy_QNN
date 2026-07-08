from __future__ import annotations

from collections.abc import Sequence

import pennylane as qml
import torch
import torch.nn as nn


class QuantumRuleCircuit(nn.Module):
    def __init__(
        self,
        n_rules: int,
        n_classes: int,
        n_layers: int,
        dev_name: str = "default.qubit",
        diff_method: str = "backprop",
        quantum_inputs_on_cpu: bool = False,
    ) -> None:
        super().__init__()
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dev_name = dev_name
        self.diff_method = diff_method
        self.quantum_inputs_on_cpu = quantum_inputs_on_cpu
        self.n_qubits = n_rules + n_classes
        self.dev = qml.device(dev_name, wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def circuit(
            alpha_single: torch.Tensor,
            theta: torch.Tensor,
            phi: torch.Tensor,
        ) -> Sequence[torch.Tensor]:
            eps = 1e-7

            for rule_index in range(self.n_rules):
                clipped_alpha = torch.clamp(alpha_single[rule_index], eps, 1.0 - eps)
                angle = 2.0 * torch.arcsin(torch.sqrt(clipped_alpha))
                qml.RY(angle, wires=rule_index)

            for layer_index in range(self.n_layers):
                for rule_index in range(self.n_rules):
                    qml.RY(theta[layer_index, rule_index, 0], wires=rule_index)
                    qml.RZ(theta[layer_index, rule_index, 1], wires=rule_index)

                for rule_index in range(self.n_rules - 1):
                    qml.CNOT(wires=[rule_index, rule_index + 1])
                if self.n_rules > 1:
                    qml.CNOT(wires=[self.n_rules - 1, 0])

                for rule_index in range(self.n_rules):
                    for class_index in range(self.n_classes):
                        qml.CRY(
                            phi[layer_index, rule_index, class_index],
                            wires=[rule_index, self.n_rules + class_index],
                        )

            return [
                qml.expval(qml.PauliZ(self.n_rules + class_index))
                for class_index in range(self.n_classes)
            ]

        self._circuit = circuit

    def forward(self, alpha: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        source_device = alpha.device
        if self.quantum_inputs_on_cpu:
            alpha = alpha.to("cpu")
            theta = theta.to("cpu")
            phi = phi.to("cpu")
        outputs: list[torch.Tensor] = []
        for alpha_single in alpha:
            circuit_output = self._circuit(alpha_single, theta, phi)
            if isinstance(circuit_output, torch.Tensor):
                outputs.append(circuit_output)
            else:
                outputs.append(torch.stack(list(circuit_output)))
        stacked_outputs = torch.stack(outputs, dim=0)
        if self.quantum_inputs_on_cpu and stacked_outputs.device != source_device:
            return stacked_outputs.to(source_device)
        return stacked_outputs

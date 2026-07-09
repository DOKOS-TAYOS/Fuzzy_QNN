from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


def inverse_softplus(value: float) -> float:
    tensor = torch.tensor(value, dtype=torch.float32)
    return torch.log(torch.expm1(tensor)).item()


class GaussianFuzzyLayer(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_fuzzy_sets: int,
        init_sigma: float = 0.15,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.n_fuzzy_sets = n_fuzzy_sets
        self.eps = eps
        centers = torch.linspace(0.0, 1.0, n_fuzzy_sets).repeat(d_in, 1)
        self.centers = nn.Parameter(centers)
        raw_sigma_value = inverse_softplus(init_sigma)
        raw_sigmas = torch.full((d_in, n_fuzzy_sets), raw_sigma_value, dtype=torch.float32)
        self.raw_sigmas = nn.Parameter(raw_sigmas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigmas = functional.softplus(self.raw_sigmas) + 1e-3
        x_expanded = x.unsqueeze(-1)
        squared_distance = (x_expanded - self.centers) ** 2
        denominator = 2.0 * (sigmas**2) + self.eps
        memberships = torch.exp(-(squared_distance / denominator))
        membership_sums = memberships.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return memberships / membership_sums

    def sigma_penalty(self) -> torch.Tensor:
        sigmas = functional.softplus(self.raw_sigmas) + 1e-3
        return torch.sum(1.0 / (sigmas**2))

    def export_parameters(self) -> dict[str, list[list[float]]]:
        sigmas = functional.softplus(self.raw_sigmas) + 1e-3
        return {
            "centers": self.centers.detach().cpu().tolist(),
            "sigmas": sigmas.detach().cpu().tolist(),
        }


class FuzzyRuleLayer(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_fuzzy_sets: int,
        n_rules: int,
        seed: int = 1234,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.n_fuzzy_sets = n_fuzzy_sets
        self.n_rules = n_rules
        self.eps = eps
        generator = torch.Generator().manual_seed(seed)
        rules = torch.randint(0, n_fuzzy_sets, (n_rules, d_in), generator=generator)
        self.register_buffer("rules", rules, persistent=True)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        batch_size = mu.shape[0]
        expanded_mu = mu.unsqueeze(1).expand(batch_size, self.n_rules, self.d_in, self.n_fuzzy_sets)
        rule_indices = self.rules.unsqueeze(0).expand(batch_size, self.n_rules, self.d_in)
        selected = torch.gather(expanded_mu, dim=3, index=rule_indices.unsqueeze(-1)).squeeze(-1)
        alpha = torch.prod(selected, dim=-1)
        alpha_sums = alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return alpha / alpha_sums

    def export_rules(self) -> list[list[int]]:
        return self.rules.detach().cpu().tolist()


class FuzzyBlock(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_fuzzy_sets: int,
        n_rules: int,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        self.fuzzy_layer: GaussianFuzzyLayer = GaussianFuzzyLayer(
            d_in=d_in, n_fuzzy_sets=n_fuzzy_sets
        )
        self.rule_layer: FuzzyRuleLayer = FuzzyRuleLayer(
            d_in=d_in,
            n_fuzzy_sets=n_fuzzy_sets,
            n_rules=n_rules,
            seed=seed,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memberships = self.fuzzy_layer(x)
        activations = self.rule_layer(memberships)
        return memberships, activations

    def sigma_penalty(self) -> torch.Tensor:
        return self.fuzzy_layer.sigma_penalty()

    def export_parameters(self) -> dict[str, object]:
        return {
            "rules": self.rule_layer.export_rules(),
            "fuzzy_layer": self.fuzzy_layer.export_parameters(),
        }

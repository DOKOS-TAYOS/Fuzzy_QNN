from __future__ import annotations

import torch

from fuzzy_qnn.fuzzy import FuzzyBlock, FuzzyRuleLayer, GaussianFuzzyLayer


def test_gaussian_fuzzy_layer_shapes_and_normalization() -> None:
    layer = GaussianFuzzyLayer(d_in=4, n_fuzzy_sets=3)
    x = torch.rand(5, 4)

    memberships = layer(x)

    assert memberships.shape == (5, 4, 3)
    assert torch.allclose(memberships.sum(dim=-1), torch.ones(5, 4), atol=1e-5)


def test_rule_layer_shapes_and_normalization() -> None:
    layer = FuzzyRuleLayer(d_in=4, n_fuzzy_sets=3, n_rules=6, seed=1234)
    memberships = torch.rand(5, 4, 3)
    memberships = memberships / memberships.sum(dim=-1, keepdim=True)

    activations = layer(memberships)

    assert activations.shape == (5, 6)
    assert torch.allclose(activations.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_fuzzy_block_returns_memberships_and_rule_activations() -> None:
    block = FuzzyBlock(d_in=4, n_fuzzy_sets=3, n_rules=6, seed=7)
    x = torch.rand(2, 4)

    memberships, activations = block(x)

    assert memberships.shape == (2, 4, 3)
    assert activations.shape == (2, 6)

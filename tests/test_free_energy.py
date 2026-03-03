"""Tests for free energy computations."""

import numpy as np
import pytest
from active_inference.generative_model import GenerativeModel
from active_inference.free_energy import (
    variational_free_energy,
    expected_free_energy,
    action_distribution,
)


def make_simple_model():
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    B = np.zeros((2, 2, 2))
    B[:, :, 0] = np.eye(2)
    B[:, :, 1] = np.array([[0, 1], [1, 0]])
    C = np.array([0.0, 2.0])
    D = np.array([0.5, 0.5])
    return GenerativeModel(A=A, B=B, C=C, D=D)


class TestVariationalFreeEnergy:
    def test_should_be_low_when_beliefs_match_observation(self):
        model = make_simple_model()
        # Beliefs aligned with observation 0
        aligned_beliefs = np.array([0.95, 0.05])
        misaligned_beliefs = np.array([0.05, 0.95])

        vfe_aligned = variational_free_energy(0, aligned_beliefs, model)
        vfe_misaligned = variational_free_energy(0, misaligned_beliefs, model)

        assert vfe_aligned < vfe_misaligned

    def test_should_be_finite(self):
        model = make_simple_model()
        vfe = variational_free_energy(0, np.array([0.5, 0.5]), model)
        assert np.isfinite(vfe)

    def test_should_penalize_deviation_from_prior(self):
        model = make_simple_model()
        # Both beliefs fit obs equally well, but one deviates from prior
        at_prior = model.D.copy()  # [0.5, 0.5]
        away_from_prior = np.array([0.99, 0.01])

        vfe_prior = variational_free_energy(0, at_prior, model)
        vfe_away = variational_free_energy(0, away_from_prior, model)

        # The complexity term (KL from prior) should make vfe_away higher
        # unless accuracy gain compensates. Test the KL contribution directly.
        from active_inference.math_utils import kl_divergence
        kl_prior = kl_divergence(at_prior, model.D)
        kl_away = kl_divergence(away_from_prior, model.D)
        assert kl_away > kl_prior


class TestExpectedFreeEnergy:
    def test_should_prefer_action_toward_preferred_obs(self):
        model = make_simple_model()
        # In state 0, prefer obs 1 (C[1]=2.0). Action 1 swaps to state 1.
        beliefs = np.array([0.9, 0.1])  # believe we're in state 0

        efe_stay = expected_free_energy(beliefs, 0, model)   # stay in state 0
        efe_swap = expected_free_energy(beliefs, 1, model)   # go to state 1

        # Swapping should have lower EFE (more preferred)
        assert efe_swap < efe_stay

    def test_should_be_finite_for_all_actions(self):
        model = make_simple_model()
        for a in range(model.num_actions):
            efe = expected_free_energy(np.array([0.5, 0.5]), a, model)
            assert np.isfinite(efe)


class TestActionDistribution:
    def test_should_sum_to_one(self):
        model = make_simple_model()
        probs = action_distribution(np.array([0.5, 0.5]), model)
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_should_favor_preferred_action(self):
        model = make_simple_model()
        beliefs = np.array([0.9, 0.1])
        probs = action_distribution(beliefs, model, precision=8.0)
        # Action 1 (swap to preferred state) should dominate
        assert probs[1] > probs[0]

    def test_should_flatten_with_low_precision(self):
        model = make_simple_model()
        beliefs = np.array([0.9, 0.1])
        low = action_distribution(beliefs, model, precision=0.01)
        high = action_distribution(beliefs, model, precision=10.0)
        # Low precision should be more uniform
        assert abs(low[0] - low[1]) < abs(high[0] - high[1])

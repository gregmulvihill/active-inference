"""Tests for the generative model."""

import numpy as np
import pytest
from active_inference.generative_model import GenerativeModel


def make_simple_model():
    """2 states, 2 observations, 2 actions."""
    A = np.array([[0.9, 0.1], [0.1, 0.9]])  # obs 0 → state 0, obs 1 → state 1
    B = np.zeros((2, 2, 2))
    B[:, :, 0] = np.eye(2)        # action 0: stay
    B[:, :, 1] = np.array([[0, 1], [1, 0]])  # action 1: swap states
    C = np.array([0.0, 2.0])      # prefer observation 1
    D = np.array([0.5, 0.5])      # uniform prior
    return GenerativeModel(A=A, B=B, C=C, D=D)


class TestGenerativeModel:
    def test_should_initialize_with_correct_dimensions(self):
        model = make_simple_model()
        assert model.num_obs == 2
        assert model.num_states == 2
        assert model.num_actions == 2

    def test_should_normalize_A_columns(self):
        A = np.array([[3.0, 1.0], [1.0, 3.0]])  # unnormalized
        B = np.zeros((2, 2, 1))
        B[:, :, 0] = np.eye(2)
        model = GenerativeModel(A=A, B=B)
        col_sums = model.A.sum(axis=0)
        np.testing.assert_allclose(col_sums, [1.0, 1.0], atol=1e-10)

    def test_predict_obs_should_return_valid_distribution(self):
        model = make_simple_model()
        beliefs = np.array([1.0, 0.0])  # certain about state 0
        pred = model.predict_obs(beliefs)
        assert abs(pred.sum() - 1.0) < 1e-10
        np.testing.assert_allclose(pred, [0.9, 0.1], atol=1e-10)

    def test_predict_next_state_should_swap_on_action_1(self):
        model = make_simple_model()
        beliefs = np.array([1.0, 0.0])
        next_state = model.predict_next_state(beliefs, action=1)
        np.testing.assert_allclose(next_state, [0.0, 1.0], atol=1e-10)

    def test_predict_next_state_should_stay_on_action_0(self):
        model = make_simple_model()
        beliefs = np.array([0.7, 0.3])
        next_state = model.predict_next_state(beliefs, action=0)
        np.testing.assert_allclose(next_state, [0.7, 0.3], atol=1e-10)

    def test_preferred_obs_should_peak_at_preferred(self):
        model = make_simple_model()
        pref = model.preferred_obs
        assert pref[1] > pref[0]  # C[1] = 2.0 > C[0] = 0.0

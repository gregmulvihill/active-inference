"""Tests for belief updating / inference."""

import numpy as np
import pytest
from active_inference.generative_model import GenerativeModel
from active_inference.inference import update_beliefs, predict_then_update


def make_simple_model():
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    B = np.zeros((2, 2, 2))
    B[:, :, 0] = np.eye(2)
    B[:, :, 1] = np.array([[0, 1], [1, 0]])
    C = np.array([0.0, 2.0])
    D = np.array([0.5, 0.5])
    return GenerativeModel(A=A, B=B, C=C, D=D)


class TestUpdateBeliefs:
    def test_should_shift_toward_observed_state(self):
        model = make_simple_model()
        prior = np.array([0.5, 0.5])

        # Observe obs 0 → should shift toward state 0
        posterior = update_beliefs(prior, obs_idx=0, model=model)
        assert posterior[0] > 0.5
        assert posterior[1] < 0.5

    def test_should_return_valid_distribution(self):
        model = make_simple_model()
        posterior = update_beliefs(np.array([0.5, 0.5]), obs_idx=1, model=model)
        assert abs(posterior.sum() - 1.0) < 1e-10
        assert np.all(posterior >= 0)

    def test_should_converge_with_repeated_evidence(self):
        model = make_simple_model()
        beliefs = np.array([0.5, 0.5])

        for _ in range(10):
            beliefs = update_beliefs(beliefs, obs_idx=0, model=model)

        # Should become very confident about state 0
        assert beliefs[0] > 0.99

    def test_should_be_uncertain_with_ambiguous_likelihood(self):
        # A matrix where observations don't distinguish states
        A = np.array([[0.5, 0.5], [0.5, 0.5]])
        B = np.zeros((2, 2, 1))
        B[:, :, 0] = np.eye(2)
        model = GenerativeModel(A=A, B=B)

        prior = np.array([0.5, 0.5])
        posterior = update_beliefs(prior, obs_idx=0, model=model)
        np.testing.assert_allclose(posterior, [0.5, 0.5], atol=1e-10)


class TestPredictThenUpdate:
    def test_should_predict_swap_then_update(self):
        model = make_simple_model()
        beliefs = np.array([0.9, 0.1])

        # Action 1 swaps, then observe obs 1 (consistent with state 1)
        posterior = predict_then_update(beliefs, action=1, obs_idx=1, model=model)

        # Should be confident about state 1
        assert posterior[1] > 0.8

    def test_should_handle_surprising_observation(self):
        model = make_simple_model()
        beliefs = np.array([0.9, 0.1])

        # Action 0 stays, but observe obs 1 (inconsistent with state 0)
        posterior = predict_then_update(beliefs, action=0, obs_idx=1, model=model)

        # Should shift beliefs toward state 1 despite prior
        assert posterior[1] > posterior[0]

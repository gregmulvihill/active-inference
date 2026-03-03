"""Tests for math utilities."""

import numpy as np
import pytest
from active_inference.math_utils import (
    softmax,
    log_stable,
    kl_divergence,
    entropy,
    normalize,
    dot_likelihood,
)


class TestSoftmax:
    def test_should_return_uniform_when_inputs_equal(self):
        result = softmax([1.0, 1.0, 1.0])
        np.testing.assert_allclose(result, [1/3, 1/3, 1/3], atol=1e-10)

    def test_should_sum_to_one(self):
        result = softmax([3.0, 1.0, 0.5])
        assert abs(result.sum() - 1.0) < 1e-10

    def test_should_sharpen_with_high_precision(self):
        low_prec = softmax([2.0, 1.0, 0.0], precision=0.5)
        high_prec = softmax([2.0, 1.0, 0.0], precision=10.0)
        # High precision should concentrate more on the max
        assert high_prec[0] > low_prec[0]

    def test_should_handle_large_values_without_overflow(self):
        result = softmax([1000.0, 999.0, 998.0])
        assert np.all(np.isfinite(result))
        assert abs(result.sum() - 1.0) < 1e-10


class TestKLDivergence:
    def test_should_return_zero_when_distributions_equal(self):
        p = np.array([0.5, 0.3, 0.2])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_should_return_positive_when_distributions_differ(self):
        q = np.array([0.9, 0.05, 0.05])
        p = np.array([1/3, 1/3, 1/3])
        assert kl_divergence(q, p) > 0

    def test_should_be_asymmetric(self):
        q = np.array([0.9, 0.1])
        p = np.array([0.5, 0.5])
        assert kl_divergence(q, p) != pytest.approx(kl_divergence(p, q))


class TestEntropy:
    def test_should_return_zero_for_deterministic(self):
        p = np.array([1.0, 0.0, 0.0])
        assert entropy(p) == pytest.approx(0.0, abs=1e-10)

    def test_should_maximize_for_uniform(self):
        n = 5
        uniform = np.ones(n) / n
        peaked = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
        assert entropy(uniform) > entropy(peaked)


class TestNormalize:
    def test_should_sum_to_one(self):
        result = normalize(np.array([3.0, 1.0, 2.0]))
        assert abs(result.sum() - 1.0) < 1e-10

    def test_should_return_uniform_for_zeros(self):
        result = normalize(np.zeros(4))
        np.testing.assert_allclose(result, np.ones(4) / 4)


class TestDotLikelihood:
    def test_should_extract_correct_row(self):
        A = np.array([[0.9, 0.1], [0.1, 0.9]])
        result = dot_likelihood(A, 0)
        np.testing.assert_allclose(result, [0.9, 0.1])

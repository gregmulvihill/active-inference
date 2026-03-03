"""Numerical utilities for active inference computations."""

import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray, precision: float = 1.0) -> NDArray:
    """Precision-weighted softmax. Higher precision = sharper distribution."""
    scaled = precision * np.array(x, dtype=np.float64)
    shifted = scaled - scaled.max()
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum()


def log_stable(x: NDArray) -> NDArray:
    """Numerically stable log — clamps near-zero values."""
    return np.log(np.maximum(x, 1e-16))


def kl_divergence(q: NDArray, p: NDArray) -> float:
    """KL divergence D_KL(q || p). Both must be proper distributions."""
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    mask = q > 1e-16
    return float(np.sum(q[mask] * (log_stable(q[mask]) - log_stable(p[mask]))))


def entropy(p: NDArray) -> float:
    """Shannon entropy H(p)."""
    p = np.asarray(p, dtype=np.float64)
    mask = p > 1e-16
    return float(-np.sum(p[mask] * log_stable(p[mask])))


def normalize(x: NDArray) -> NDArray:
    """Normalize array to sum to 1. Returns uniform if all zeros."""
    x = np.asarray(x, dtype=np.float64)
    total = x.sum()
    if total < 1e-16:
        return np.ones_like(x) / len(x)
    return x / total


def dot_likelihood(A: NDArray, obs_idx: int) -> NDArray:
    """Extract likelihood of observation given each hidden state.

    A[o, s] = P(observation=o | state=s)
    Returns A[obs_idx, :] — the likelihood vector for the observed outcome.
    """
    return A[obs_idx, :]

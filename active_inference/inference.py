"""Belief updating — perception as inference under uncertainty.

The agent doesn't directly perceive the world. It receives observations
through a sensory boundary (Markov blanket) and must infer the hidden
state of the world using Bayesian updating.

This implements the "predict, compare, update" loop from the transcript:
1. Predict: what do I expect to observe given my beliefs?
2. Compare: what did I actually observe?
3. Update: revise beliefs to reduce prediction error.
"""

import numpy as np
from numpy.typing import NDArray

from active_inference.math_utils import normalize, log_stable, dot_likelihood
from active_inference.generative_model import GenerativeModel


def update_beliefs(
    prior: NDArray,
    obs_idx: int,
    model: GenerativeModel,
    num_iters: int = 16,
) -> NDArray:
    """Update beliefs about hidden states given a new observation.

    Uses iterative fixed-point Bayesian inference:
        q(s) ∝ P(o|s) * prior(s)

    For discrete state spaces, this is exact in one step. The iterative
    form generalizes to deeper models with multiple levels.

    Parameters
    ----------
    prior : (num_states,) array
        Prior beliefs about hidden states (before observation).
    obs_idx : int
        Index of the observed outcome.
    model : GenerativeModel
        The agent's generative model.
    num_iters : int
        Number of belief update iterations (1 is sufficient for single-level).

    Returns
    -------
    (num_states,) array
        Posterior beliefs about hidden states.
    """
    prior = np.asarray(prior, dtype=np.float64)
    likelihood = dot_likelihood(model.A, obs_idx)

    # Bayes rule: posterior ∝ likelihood * prior
    posterior = normalize(likelihood * prior)

    # Iterative refinement (for future hierarchical extensions)
    for _ in range(num_iters - 1):
        posterior = normalize(likelihood * posterior)

    return posterior


def predict_then_update(
    beliefs: NDArray,
    action: int,
    obs_idx: int,
    model: GenerativeModel,
) -> NDArray:
    """Full perception cycle: predict next state, then update with observation.

    This is the core "predict, compare, update" loop:
    1. Use transition model to predict next state: P(s'|s, a)
    2. Use likelihood model to update with observation: P(s'|o, a)

    Parameters
    ----------
    beliefs : (num_states,) array
        Current beliefs about hidden states.
    action : int
        Action that was taken.
    obs_idx : int
        Observation received after acting.
    model : GenerativeModel
        The agent's generative model.

    Returns
    -------
    (num_states,) array
        Updated posterior beliefs.
    """
    # Predict: what state do I expect after this action?
    predicted = model.predict_next_state(beliefs, action)

    # Update: revise prediction given actual observation
    posterior = update_beliefs(predicted, obs_idx, model)

    return posterior

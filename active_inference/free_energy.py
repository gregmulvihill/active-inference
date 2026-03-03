"""Free energy computations — the core optimization target of active inference.

Variational Free Energy (VFE):
    Measures how badly the agent's beliefs fit its observations.
    F = E_q[log q(s) - log p(o,s)] = KL(q(s) || p(s)) - log p(o)
    Minimizing F means: improve your model or change your beliefs.

Expected Free Energy (EFE):
    Evaluates future actions by balancing:
    - Pragmatic value (will this action lead to preferred outcomes?)
    - Epistemic value (will this action reduce uncertainty?)
    G(a) = pragmatic_term + epistemic_term
    This naturally resolves exploration vs exploitation.
"""

import numpy as np
from numpy.typing import NDArray

from active_inference.math_utils import (
    log_stable,
    kl_divergence,
    entropy,
    normalize,
)
from active_inference.generative_model import GenerativeModel


def variational_free_energy(
    obs_idx: int,
    beliefs: NDArray,
    model: GenerativeModel,
    empirical_prior: NDArray | None = None,
) -> float:
    """Compute variational free energy given observation and current beliefs.

    F = -E_q[log P(o|s)] + KL(q(s) || prior(s))

    Lower F means better fit between beliefs and observations.

    Parameters
    ----------
    obs_idx : int
        Index of the observed outcome.
    beliefs : (num_states,) array
        Current posterior beliefs about hidden states q(s).
    model : GenerativeModel
        The agent's generative model.
    empirical_prior : (num_states,) array or None
        Empirical prior from transition prediction. If None, uses model.D.

    Returns
    -------
    float
        Variational free energy value.
    """
    beliefs = np.asarray(beliefs, dtype=np.float64)
    prior = empirical_prior if empirical_prior is not None else model.D

    # Accuracy: expected log-likelihood of observation under beliefs
    # E_q[log P(o|s)] = sum_s q(s) * log A[o, s]
    log_likelihood = log_stable(model.A[obs_idx, :])
    accuracy = float(np.dot(beliefs, log_likelihood))

    # Complexity: KL divergence from prior (empirical or initial)
    complexity = kl_divergence(beliefs, prior)

    return -accuracy + complexity


def expected_free_energy(
    beliefs: NDArray,
    action: int,
    model: GenerativeModel,
) -> float:
    """Compute expected free energy for a candidate action.

    G(a) = pragmatic_value + epistemic_value

    Pragmatic: KL(predicted_obs || preferred_obs) — does this action
               lead to outcomes I prefer?
    Epistemic: -expected_info_gain — does this action reduce my
               uncertainty about hidden states?

    Lower G means the action is more desirable (preferred outcomes + learning).

    Parameters
    ----------
    beliefs : (num_states,) array
        Current beliefs about hidden states.
    action : int
        Candidate action index.
    model : GenerativeModel
        The agent's generative model.

    Returns
    -------
    float
        Expected free energy for the action.
    """
    beliefs = np.asarray(beliefs, dtype=np.float64)

    # Predict next state distribution under this action
    predicted_states = model.predict_next_state(beliefs, action)

    # Predict what we'd observe from those states
    predicted_obs = model.predict_obs(predicted_states)

    # --- Pragmatic value ---
    # How far are predicted observations from preferred observations?
    # KL(predicted_obs || preferred_obs) — want this low
    preferred = model.preferred_obs
    pragmatic = kl_divergence(predicted_obs, preferred)

    # --- Epistemic value ---
    # Expected information gain: how much would observations reduce
    # uncertainty about hidden states?
    # H(s|o) - approximated as expected conditional entropy
    # Higher info gain (lower epistemic term) = more valuable for learning
    epistemic = 0.0
    for s_idx in range(model.num_states):
        if predicted_states[s_idx] < 1e-16:
            continue
        # Ambiguity: entropy of observation distribution given this state
        obs_given_state = model.A[:, s_idx]
        epistemic += predicted_states[s_idx] * entropy(obs_given_state)

    return pragmatic + epistemic


def action_distribution(
    beliefs: NDArray,
    model: GenerativeModel,
    precision: float = 1.0,
) -> NDArray:
    """Compute action probabilities via expected free energy.

    P(a) = softmax(-precision * G(a))

    Actions with lower expected free energy are more probable.
    Precision controls how deterministic the policy is:
    - High precision → exploit best action
    - Low precision → explore more uniformly

    Parameters
    ----------
    beliefs : (num_states,) array
        Current beliefs about hidden states.
    model : GenerativeModel
        The agent's generative model.
    precision : float
        Action precision (inverse temperature). Higher = more decisive.

    Returns
    -------
    (num_actions,) array
        Probability distribution over actions.
    """
    from active_inference.math_utils import softmax

    G = np.array([
        expected_free_energy(beliefs, a, model)
        for a in range(model.num_actions)
    ])

    # Negate because we want to MINIMIZE free energy
    return softmax(-G, precision=precision)

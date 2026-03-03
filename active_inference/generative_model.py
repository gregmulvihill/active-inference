"""Generative model for active inference in discrete state spaces.

A generative model encodes an agent's beliefs about how the world works:
- How hidden states produce observations (A matrix — likelihood)
- How states transition given actions (B matrix — dynamics)
- What observations the agent prefers (C vector — preferences)
- What the agent believes about initial states (D vector — prior)

This is the agent's "internal scientist" — it can simulate what it expects
to observe if the world is in a given state and it takes a given action.
"""

import numpy as np
from numpy.typing import NDArray

from active_inference.math_utils import normalize


class GenerativeModel:
    """Discrete-state generative model for active inference.

    Parameters
    ----------
    A : (num_obs, num_states) array
        Likelihood mapping. A[o, s] = P(observation=o | state=s).
    B : (num_states, num_states, num_actions) array
        Transition dynamics. B[s', s, a] = P(next_state=s' | state=s, action=a).
    C : (num_obs,) array
        Log-preference over observations. Higher = more preferred.
    D : (num_states,) array
        Prior beliefs about initial hidden state.
    """

    def __init__(
        self,
        A: NDArray,
        B: NDArray,
        C: NDArray | None = None,
        D: NDArray | None = None,
    ):
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)

        self.num_obs, self.num_states = self.A.shape
        self.num_actions = self.B.shape[2]

        self._validate_matrices()

        if C is not None:
            self.C = np.asarray(C, dtype=np.float64)
        else:
            self.C = np.zeros(self.num_obs)

        if D is not None:
            self.D = normalize(np.asarray(D, dtype=np.float64))
        else:
            self.D = np.ones(self.num_states) / self.num_states

    def _validate_matrices(self):
        """Check matrix dimensions and normalization."""
        assert self.A.ndim == 2, f"A must be 2D, got {self.A.ndim}D"
        assert self.B.ndim == 3, f"B must be 3D, got {self.B.ndim}D"
        assert self.B.shape[0] == self.num_states, "B dim 0 must match num_states"
        assert self.B.shape[1] == self.num_states, "B dim 1 must match num_states"

        # A columns should sum to 1 (each state generates a valid obs distribution)
        col_sums = self.A.sum(axis=0)
        if not np.allclose(col_sums, 1.0, atol=1e-6):
            self.A = self.A / col_sums[np.newaxis, :]

        # B columns should sum to 1 for each action
        for a in range(self.num_actions):
            col_sums = self.B[:, :, a].sum(axis=0)
            if not np.allclose(col_sums, 1.0, atol=1e-6):
                self.B[:, :, a] = self.B[:, :, a] / col_sums[np.newaxis, :]

    def predict_obs(self, beliefs: NDArray) -> NDArray:
        """Predict observation distribution given beliefs about hidden states.

        Returns P(o) = sum_s A[o,s] * beliefs[s]
        """
        return self.A @ beliefs

    def predict_next_state(self, beliefs: NDArray, action: int) -> NDArray:
        """Predict next state distribution given current beliefs and action.

        Returns P(s') = sum_s B[s', s, action] * beliefs[s]
        """
        return self.B[:, :, action] @ beliefs

    @property
    def preferred_obs(self) -> NDArray:
        """Softmax of C — the preferred observation distribution."""
        from active_inference.math_utils import softmax
        return softmax(self.C)

"""Active inference agent — the complete perception-action loop.

Brings together:
- Generative model (world knowledge)
- Belief updating (perception as inference)
- Action selection via expected free energy (planning)
- Precision weighting (attention / confidence)

The agent runs the loop: observe → infer state → select action → act → repeat.
"""

import numpy as np
from numpy.typing import NDArray

from active_inference.generative_model import GenerativeModel
from active_inference.inference import update_beliefs, predict_then_update
from active_inference.free_energy import (
    variational_free_energy,
    expected_free_energy,
    action_distribution,
)
from active_inference.math_utils import entropy


class ActiveInferenceAgent:
    """Agent that uses active inference for perception and action.

    Parameters
    ----------
    model : GenerativeModel
        The agent's generative model of the world.
    precision : float
        Action precision (inverse temperature). Controls exploration:
        high = exploit, low = explore.
    precision_learning_rate : float
        Rate at which precision adapts based on prediction accuracy.
        Set to 0 to disable adaptive precision.
    """

    def __init__(
        self,
        model: GenerativeModel,
        precision: float = 1.0,
        precision_learning_rate: float = 0.0,
    ):
        self.model = model
        self.precision = precision
        self.precision_lr = precision_learning_rate

        # Initialize beliefs to prior
        self.beliefs: NDArray = model.D.copy()
        self._empirical_prior: NDArray | None = None  # predicted state before observation

        # Tracking
        self.history: list[dict] = []
        self.step_count: int = 0

    def reset(self):
        """Reset agent to initial beliefs."""
        self.beliefs = self.model.D.copy()
        self._empirical_prior = None
        self.history = []
        self.step_count = 0

    def observe(self, obs_idx: int) -> NDArray:
        """Update beliefs given a new observation (perception step).

        Returns updated posterior beliefs.
        """
        self.beliefs = update_beliefs(self.beliefs, obs_idx, self.model)
        return self.beliefs

    def act(self) -> int:
        """Select an action by evaluating expected free energy.

        Returns the selected action index.
        """
        # Compute action probabilities
        action_probs = action_distribution(
            self.beliefs, self.model, precision=self.precision
        )

        # Sample action from distribution
        action = int(np.random.choice(self.model.num_actions, p=action_probs))

        return action

    def step(self, obs_idx: int, prev_action: int | None = None) -> int:
        """Full perception-action cycle.

        1. Update beliefs with observation (and previous action if given)
        2. Compute free energy (model fit)
        3. Select next action
        4. Optionally adapt precision

        Parameters
        ----------
        obs_idx : int
            Current observation.
        prev_action : int or None
            Action taken before this observation (for transition-aware update).

        Returns
        -------
        int
            Selected action.
        """
        # --- Perception ---
        if prev_action is not None and self.step_count > 0:
            # Track empirical prior (predicted state before observation)
            self._empirical_prior = self.model.predict_next_state(
                self.beliefs, prev_action
            )
            self.beliefs = predict_then_update(
                self.beliefs, prev_action, obs_idx, self.model
            )
        else:
            self._empirical_prior = None
            self.observe(obs_idx)

        # --- Evaluate model fit ---
        # Use empirical prior (predicted state) instead of fixed D
        vfe = variational_free_energy(
            obs_idx, self.beliefs, self.model,
            empirical_prior=self._empirical_prior,
        )

        # --- Action selection ---
        action_probs = action_distribution(
            self.beliefs, self.model, precision=self.precision
        )
        action = int(np.random.choice(self.model.num_actions, p=action_probs))

        # --- EFE for all actions (for logging) ---
        efe_values = np.array([
            expected_free_energy(self.beliefs, a, self.model)
            for a in range(self.model.num_actions)
        ])

        # --- Adaptive precision ---
        if self.precision_lr > 0:
            self._adapt_precision(vfe)

        # --- Record ---
        record = {
            "step": self.step_count,
            "obs": obs_idx,
            "beliefs": self.beliefs.copy(),
            "vfe": vfe,
            "efe": efe_values.copy(),
            "action_probs": action_probs.copy(),
            "action": action,
            "precision": self.precision,
            "belief_entropy": entropy(self.beliefs),
        }
        self.history.append(record)
        self.step_count += 1

        return action

    def _adapt_precision(self, vfe: float):
        """Adapt precision based on free energy.

        Low free energy (good predictions) → increase precision (exploit more).
        High free energy (poor predictions) → decrease precision (explore more).
        """
        # Simple gradient: precision moves opposite to free energy
        # Clamped to [0.1, 32.0] for numerical stability
        self.precision = np.clip(
            self.precision - self.precision_lr * (vfe - 1.0),
            0.1,
            32.0,
        )

    def report(self) -> dict:
        """Summary of agent's current state."""
        return {
            "step": self.step_count,
            "beliefs": self.beliefs.copy(),
            "belief_entropy": entropy(self.beliefs),
            "precision": self.precision,
            "last_vfe": self.history[-1]["vfe"] if self.history else None,
        }

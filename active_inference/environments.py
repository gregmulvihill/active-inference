"""Environments for testing active inference agents.

Each environment is a partially observable Markov decision process (POMDP).
The agent cannot see the true state — it receives noisy observations and
must infer what's happening through the Markov blanket.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class StepResult:
    obs: int
    reward: float
    done: bool
    info: dict


class GridWorldEnv:
    """Simple 1D grid world with a hidden reward location.

    The agent moves left/right on a 1D grid, receives noisy position
    observations, and gets reward at a target location.

    This demonstrates:
    - Perception under uncertainty (noisy observations)
    - Exploration vs exploitation (find the target vs stay at it)
    - Markov blanket (agent only sees observations, not true state)

    States: positions [0, 1, ..., grid_size-1]
    Actions: 0=stay, 1=move_left, 2=move_right
    Observations: noisy version of position (can be wrong)
    """

    def __init__(
        self,
        grid_size: int = 5,
        target: int | None = None,
        obs_noise: float = 0.1,
    ):
        self.grid_size = grid_size
        self.target = target if target is not None else grid_size - 1
        self.obs_noise = obs_noise

        self.state: int = 0
        self.step_count: int = 0

    def reset(self, start: int = 0) -> int:
        """Reset environment. Returns initial observation."""
        self.state = start
        self.step_count = 0
        return self._observe()

    def step(self, action: int) -> StepResult:
        """Take action, return observation.

        Actions: 0=stay, 1=left, 2=right
        """
        # Transition
        if action == 1:  # left
            self.state = max(0, self.state - 1)
        elif action == 2:  # right
            self.state = min(self.grid_size - 1, self.state + 1)
        # action 0 = stay

        self.step_count += 1

        obs = self._observe()
        reward = 1.0 if self.state == self.target else 0.0
        done = self.step_count >= 50

        return StepResult(
            obs=obs,
            reward=reward,
            done=done,
            info={"true_state": self.state, "target": self.target},
        )

    def _observe(self) -> int:
        """Generate noisy observation of current state."""
        if np.random.random() < self.obs_noise:
            # Wrong observation — uniform over other positions
            others = [i for i in range(self.grid_size) if i != self.state]
            return int(np.random.choice(others))
        return self.state

    def build_generative_model(self):
        """Build the ground-truth generative model matrices for this environment.

        Returns (A, B, C, D) matrices suitable for GenerativeModel.
        """
        n = self.grid_size

        # A: observation likelihood. A[o, s] = P(obs=o | state=s)
        A = np.full((n, n), self.obs_noise / (n - 1))
        np.fill_diagonal(A, 1.0 - self.obs_noise)

        # B: transition dynamics. B[s', s, a] = P(next=s' | current=s, action=a)
        B = np.zeros((n, n, 3))

        for s in range(n):
            # Action 0: stay
            B[s, s, 0] = 1.0

            # Action 1: left
            s_left = max(0, s - 1)
            B[s_left, s, 1] = 1.0

            # Action 2: right
            s_right = min(n - 1, s + 1)
            B[s_right, s, 2] = 1.0

        # C: preferences over observations (log scale)
        # Strongly prefer observing the target location
        C = np.zeros(n)
        C[self.target] = 4.0  # log-preference for target

        # D: initial state prior (start at position 0)
        D = np.zeros(n)
        D[0] = 1.0

        return A, B, C, D


class TMazeEnv:
    """T-maze environment — classic active inference benchmark.

    The agent starts at the bottom of a T. It must go up to the junction,
    then choose left or right. One arm has reward, the other doesn't.
    A cue at the start hints which arm has reward.

    This demonstrates:
    - Epistemic action (seeking information before committing)
    - The exploration-exploitation balance
    - How active inference naturally seeks cues to reduce uncertainty

    States: 4 positions × 2 reward locations = 8 hidden states
        Positions: 0=start, 1=junction, 2=left_arm, 3=right_arm
        Reward context: 0=reward_left, 1=reward_right
    Actions: 0=stay, 1=go_up, 2=go_left, 3=go_right
    Observations: 6 possible
        0=at_start, 1=at_junction, 2=reward, 3=no_reward, 4=cue_left, 5=cue_right
    """

    POSITIONS = ["start", "junction", "left_arm", "right_arm"]
    REWARD_CONTEXTS = ["reward_left", "reward_right"]
    OBS_NAMES = ["at_start", "at_junction", "reward", "no_reward", "cue_left", "cue_right"]

    def __init__(self, cue_reliability: float = 0.9):
        self.cue_reliability = cue_reliability
        self.reward_context: int = 0  # 0=left, 1=right
        self.position: int = 0
        self.step_count: int = 0

    def _state_idx(self, pos: int, ctx: int) -> int:
        return pos * 2 + ctx

    def reset(self) -> int:
        """Reset. Randomly set reward context. Returns initial observation."""
        self.reward_context = int(np.random.choice(2))
        self.position = 0
        self.step_count = 0
        return self._observe()

    def step(self, action: int) -> StepResult:
        """Take action in T-maze."""
        # Transitions depend on current position
        if self.position == 0:  # start
            if action == 1:  # go_up
                self.position = 1
        elif self.position == 1:  # junction
            if action == 2:  # go_left
                self.position = 2
            elif action == 3:  # go_right
                self.position = 3
        # Arms and stay are absorbing

        self.step_count += 1
        obs = self._observe()

        # Reward only at arms
        reward = 0.0
        if self.position == 2:  # left arm
            reward = 1.0 if self.reward_context == 0 else -1.0
        elif self.position == 3:  # right arm
            reward = 1.0 if self.reward_context == 1 else -1.0

        done = self.position in (2, 3) or self.step_count >= 10

        return StepResult(
            obs=obs,
            reward=reward,
            done=done,
            info={
                "position": self.POSITIONS[self.position],
                "reward_context": self.REWARD_CONTEXTS[self.reward_context],
            },
        )

    def _observe(self) -> int:
        """Generate observation based on position and reward context."""
        if self.position == 0:
            # At start — receive cue about reward location
            if np.random.random() < self.cue_reliability:
                return 4 if self.reward_context == 0 else 5  # correct cue
            else:
                return 5 if self.reward_context == 0 else 4  # misleading cue
        elif self.position == 1:
            return 1  # at junction
        elif self.position == 2:
            return 2 if self.reward_context == 0 else 3  # reward or not
        elif self.position == 3:
            return 2 if self.reward_context == 1 else 3
        return 0

    def build_generative_model(self):
        """Build generative model matrices for the T-maze.

        8 hidden states (4 positions × 2 contexts), 6 observations, 4 actions.
        """
        ns = 8  # states
        no = 6  # observations
        na = 4  # actions

        # --- A matrix: observation likelihood ---
        A = np.zeros((no, ns))
        r = self.cue_reliability

        for ctx in range(2):
            # Start position: cue observation
            s = self._state_idx(0, ctx)
            if ctx == 0:  # reward_left
                A[4, s] = r       # cue_left (correct)
                A[5, s] = 1 - r   # cue_right (incorrect)
            else:  # reward_right
                A[5, s] = r       # cue_right (correct)
                A[4, s] = 1 - r   # cue_left (incorrect)

            # Junction: just see junction
            s = self._state_idx(1, ctx)
            A[1, s] = 1.0

            # Left arm
            s = self._state_idx(2, ctx)
            A[2, s] = 1.0 if ctx == 0 else 0.0  # reward if ctx=left
            A[3, s] = 0.0 if ctx == 0 else 1.0  # no reward if ctx=right

            # Right arm
            s = self._state_idx(3, ctx)
            A[2, s] = 1.0 if ctx == 1 else 0.0
            A[3, s] = 0.0 if ctx == 1 else 1.0

        # --- B matrix: transitions ---
        B = np.zeros((ns, ns, na))

        for ctx in range(2):
            for pos in range(4):
                s = self._state_idx(pos, ctx)

                # Action 0: stay — context preserved
                B[s, s, 0] = 1.0

                # Action 1: go_up (only from start→junction)
                if pos == 0:
                    B[self._state_idx(1, ctx), s, 1] = 1.0
                else:
                    B[s, s, 1] = 1.0  # stay if invalid

                # Action 2: go_left (only from junction→left_arm)
                if pos == 1:
                    B[self._state_idx(2, ctx), s, 2] = 1.0
                else:
                    B[s, s, 2] = 1.0

                # Action 3: go_right (only from junction→right_arm)
                if pos == 1:
                    B[self._state_idx(3, ctx), s, 3] = 1.0
                else:
                    B[s, s, 3] = 1.0

        # --- C: prefer reward observation ---
        C = np.zeros(no)
        C[2] = 4.0   # strongly prefer seeing reward
        C[3] = -4.0  # strongly avoid seeing no_reward

        # --- D: prior — at start, uncertain about context ---
        D = np.zeros(ns)
        D[self._state_idx(0, 0)] = 0.5
        D[self._state_idx(0, 1)] = 0.5

        return A, B, C, D

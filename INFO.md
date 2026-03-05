# Active Inference Engine

## What Is This?

An AI decision-making framework based on Karl Friston's Free Energy Principle from neuroscience. The core idea: an agent has a mental model of the world and acts to minimize *surprise* — the gap between what it expects and what it observes.

This is a from-scratch Python implementation of discrete active inference (POMDP) for the Cogent Echo AI research lab. It's an alternative to reinforcement learning where the agent is driven by curiosity and surprise-minimization rather than reward maximization.

## How It Works

1. **Generative Model** — The agent has a probabilistic model of how the world works (state transitions, expected observations per state)
2. **Belief Updating** — When the agent observes something, it updates its beliefs about what state it's in (Bayesian inference)
3. **Free Energy Minimization** — The agent evaluates possible actions by predicting: "if I do X, how surprised will I be?" It picks actions that reduce expected surprise
4. **Perception-Action Loop** — Observe -> update beliefs -> evaluate actions -> act -> repeat

## Concrete Example: T-Maze

- A rat is at the start of a T-shaped maze
- It can go left or right at the junction
- One side has reward, the other doesn't
- There's a cue that hints which side has the reward
- The agent learns to read the cue and go to the correct side
- Result: 87% accuracy vs 50% random chance

## Tech Stack

- **Language:** Python 3.11+
- **Dependencies:** numpy, scipy
- **Testing:** pytest (41 tests, all passing)
- **Deployment:** Docker (multi-stage: test + demo targets)

## Module Map

| Module | Purpose |
|--------|---------|
| `math_utils.py` | softmax, log_stable, KL divergence, entropy, normalize |
| `generative_model.py` | A/B/C/D matrices, predict_obs, predict_next_state |
| `free_energy.py` | VFE (perception), EFE (action), action_distribution |
| `inference.py` | Bayesian belief updating, predict-then-update cycle |
| `agent.py` | ActiveInferenceAgent — full perception-action loop |
| `environments.py` | GridWorldEnv, TMazeEnv — POMDP test environments |

## Quick Start

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
python -m examples.grid_world_demo
python -m examples.tmaze_demo
```

## Status

- Core engine complete (5 modules, 2 environments, 2 demos)
- 41 tests passing
- Docker FC criteria met
- Grid world demo variable across random seeds (known issue)
- Continuous state space extension not yet scoped

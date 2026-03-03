# Active Inference

Discrete-state active inference engine built from first principles. Implements the free energy principle for perception (variational free energy minimization) and action (expected free energy minimization) in partially observable Markov decision processes.

## What It Does

An agent maintains beliefs about hidden world states and selects actions that jointly:
- **Reduce uncertainty** (epistemic value — curiosity/exploration)
- **Achieve preferred outcomes** (pragmatic value — goal-directed behavior)

This naturally resolves the exploration-exploitation trade-off without separate mechanisms.

## Install

```bash
pip install -e ".[dev]"
```

## Run Tests

```bash
python -m pytest tests/ -v
```

41 tests covering math utilities, generative model, free energy, belief updating, and agent behavior.

## Run Demos

```bash
python -m examples.grid_world_demo    # 1D grid navigation
python -m examples.tmaze_demo          # T-maze benchmark (87% accuracy)
```

## Docker

```bash
docker compose run test          # Run test suite
docker compose run grid-demo     # Grid world demo
docker compose run tmaze-demo    # T-maze demo
```

## Architecture

```
active_inference/
  math_utils.py         Numerical primitives (softmax, KL, entropy)
  generative_model.py   A/B/C/D matrices — the agent's world model
  free_energy.py        VFE (perception objective) + EFE (action objective)
  inference.py          Bayesian belief updating through Markov blanket
  agent.py              Perception-action loop with adaptive precision
  environments.py       GridWorld and T-maze POMDP environments
```

**Core loop:** observe → update beliefs (minimize VFE) → evaluate actions (minimize EFE) → act → repeat

## Key Concepts

| Concept | Implementation |
|---------|---------------|
| Generative model | `GenerativeModel` with A (likelihood), B (transitions), C (preferences), D (prior) |
| Variational free energy | Accuracy - Complexity decomposition. Drives perception. |
| Expected free energy | Pragmatic + Epistemic value. Drives action selection. |
| Precision | Inverse temperature on action softmax. Controls explore/exploit. |
| Markov blanket | Agent observes through A matrix, acts through B matrix. Never sees true state. |

## Environments

**GridWorld** — 1D grid (5 positions), noisy observations, target at position 4. Tests goal-directed navigation under perceptual uncertainty.

**T-maze** — Classic active inference benchmark. Agent starts at bottom of T, receives a cue about which arm has reward, must navigate to the correct arm. Tests epistemic action (seeking information before committing).

## References

- Friston, K. (2010). "The free-energy principle: a unified brain theory?" Nature Reviews Neuroscience.
- Parr, Pezzulo & Friston (2022). "Active Inference: The Free Energy Principle in Mind, Brain, and Behavior."
- Da Costa et al. (2020). "Active inference on discrete state-spaces."

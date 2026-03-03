# Active Inference — Claude Code Instructions

## Repository Purpose

Working implementation of the active inference framework (discrete POMDP) for the Cogent Echo AI research lab. Built from first principles based on the free energy principle (Friston, 2010).

## Project Type

- **Language**: Python 3.11+
- **Architecture**: Active inference engine — generative models, belief updating, free energy minimization, action selection
- **Dependencies**: numpy, scipy (see pyproject.toml)
- **Testing**: pytest, 41 tests, all passing
- **Deployment**: Docker (multi-stage: test + demo targets)

## Project Status

**Phase:** Core engine built. Tuning and extension phase.

- 5 core modules in `active_inference/`
- 2 environments: GridWorld (1D, 5 positions), T-maze (classic benchmark)
- 2 demo scripts in `examples/`
- T-maze verified: 87% accuracy vs 50% random baseline (cue=0.9)
- Grid world: passes test (seed=42), demo performance variable across seeds

## Quick Reference

```bash
# Install
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run demos
python -m examples.grid_world_demo
python -m examples.tmaze_demo

# Docker
docker compose run test
docker compose run grid-demo
docker compose run tmaze-demo
```

## Module Map

| Module | Purpose |
|--------|---------|
| `math_utils.py` | softmax, log_stable, KL divergence, entropy, normalize |
| `generative_model.py` | A/B/C/D matrices, predict_obs, predict_next_state |
| `free_energy.py` | VFE (perception), EFE (action), action_distribution |
| `inference.py` | Bayesian belief updating, predict-then-update cycle |
| `agent.py` | ActiveInferenceAgent — full perception-action loop |
| `environments.py` | GridWorldEnv, TMazeEnv — POMDP test environments |

## Core Development Mandates

### Communication & Approach (from core-mandate.md)
- **Role**: Peer-level, objective strategic coordinator (Orchestrate-AI)
- **Challenge Mandate**: Critically analyze all claims, push back when justified
- **Communication**: Conclusions first, details second. Terse but complete.
- **Verification**: Label all unverified content: [Inference], [Speculation], [Unverified]

### Technical Standards (from technical-mandate.md)
- **TDD**: Tests before implementation. 80% coverage critical paths.
- **Git**: Commit every 15-30 min. `type(scope): description` format.
- **Docker**: Multi-stage builds. Never localhost for container-to-container.
- **Documentation**: dev-docs/ for working docs per development-construct.md.
- **FC Definition**: Docker + Manual Validation Protocol (MVP).

### Prohibited Terms (unless quoting)
Prevent, Guarantee, Will never, Fixes, Eliminates, Ensures

# Manual Validation Protocol — Active Inference

Step-by-step instructions for independently testing the active inference engine.

## Prerequisites

- Python 3.11+ installed
- Docker and docker compose installed (for container tests)

## 1. Install and Run Tests

```bash
cd /path/to/active-inference
pip install -e ".[dev]"
python -m pytest tests/ -v
```

**Expected:** 41 tests pass. Zero failures.

## 2. Run T-maze Demo

```bash
python -m examples.tmaze_demo
```

**Expected:** Agent achieves >60% accuracy across 50 trials at cue_reliability=0.9. Typical result: ~87%. Output shows per-trial results and summary statistics.

## 3. Run Grid World Demo

```bash
python -m examples.grid_world_demo
```

**Expected:** Agent navigates toward target position 4 on a 5-position grid. Single-run output shows step-by-step beliefs and actions. Precision comparison shows performance across trials.

## 4. Docker — Run Tests

```bash
docker compose run test
```

**Expected:** Container builds, runs pytest, 41 tests pass.

## 5. Docker — Run Demos

```bash
docker compose run tmaze-demo
docker compose run grid-demo
```

**Expected:** Each demo runs inside the container and produces output matching steps 2-3.

## 6. Verify Agent Behavior

```python
import numpy as np
from active_inference.environments import TMazeEnv
from active_inference.generative_model import GenerativeModel
from active_inference.agent import ActiveInferenceAgent

np.random.seed(42)
env = TMazeEnv(cue_reliability=0.9)
A, B, C, D = env.build_generative_model()
model = GenerativeModel(A=A, B=B, C=C, D=D)
agent = ActiveInferenceAgent(model=model, precision=4.0)

obs = env.reset()
action = None
for step in range(10):
    action = agent.step(obs, prev_action=action)
    result = env.step(action)
    obs = result.obs
    print(f"Step {step}: action={action}, obs={result.obs}, done={result.done}")
    if result.done:
        break

print(f"Reward: {result.reward}")
```

**Expected:** Agent moves from start → junction → correct arm based on cue. Reward should be positive (1.0).

## Pass Criteria

All 6 steps produce expected results without errors.

# Architecture — Active Inference Engine

## Design

Discrete-state POMDP active inference with modular separation of concerns:

```
                    ┌──────────────────┐
                    │  Environment     │
                    │  (GridWorld /    │
                    │   TMaze)         │
                    └───────┬──────────┘
                            │ obs_idx
                    ┌───────▼──────────┐
                    │  Agent           │
                    │  (agent.py)      │
                    │                  │
                    │  ┌────────────┐  │
           obs ───► │  │ Inference  │  │
                    │  │ (Bayes)    │  │
                    │  └─────┬──────┘  │
                    │        │beliefs  │
                    │  ┌─────▼──────┐  │
                    │  │ Free       │  │
                    │  │ Energy     │  │
                    │  │ (VFE+EFE)  │  │
                    │  └─────┬──────┘  │
                    │        │action   │
                    └────────┼─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Environment     │
                    │  step(action)    │
                    └──────────────────┘
```

## Module Dependencies

```
math_utils  ← generative_model ← free_energy ← agent
                    ↑                    ↑        ↑
              inference ─────────────────┘        │
                                                  │
              environments ───────────────────────┘
```

## Key Design Decisions

1. **Discrete state spaces only** — no neural network approximators. Keeps the math exact and interpretable.
2. **Empirical prior for VFE** — after the first step, VFE uses the predicted state (from transition model) as prior instead of the fixed D vector. Reduces spurious complexity penalties when the agent moves.
3. **EMA-smoothed precision adaptation** — precision adjusts relative to a running average of VFE, not a fixed threshold. More stable across environments with different VFE ranges.
4. **Environment provides generative model** — `build_generative_model()` returns ground-truth A/B/C/D matrices. The agent uses these as its internal model (perfect model assumption for now).

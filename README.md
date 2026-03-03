# Active Inference Evaluation

Evaluation of the active inference framework for potential adoption into the Cogent Echo AI portfolio.

## Verdict

**Adopt as design philosophy. Do not adopt as software dependency.**

Active inference (Friston, 2010) provides a coherent vocabulary for agent boundary design, uncertainty management, and self-monitoring. However, it is mathematically equivalent to reinforcement learning when implemented at scale, offers no proven performance advantages, and has no production-ready tooling.

## Key Findings

| Question | Answer |
|----------|--------|
| Is active inference different from RL? | Theoretically yes, practically no (when using neural network approximators) |
| Is pymdp production-ready? | No. Discrete state spaces only, v0.0.7.1, combinatorial explosion at scale |
| Are there production deployments? | No. Research and toy demos only |
| Does the exploration bonus (EFE) work? | Derivation is contested. Standard curiosity-driven RL achieves similar results |
| Is the LLM + active inference hybrid viable? | [Speculation] No implementations exist. Worth monitoring |

## What's Valuable for Cogent Echo

| Concept | Application |
|---------|-------------|
| Markov blankets | Formal agent boundary definition |
| Precision weighting | Attention allocation / signal trust |
| Free energy as prediction error | Agent self-monitoring health metric |
| Generative models | Explicit, introspectable world models |

## Structure

- `ARTIFACT-BASELINE.md` — Full evaluation artifact with decision matrix
- `analysis/` — Content analysis reports
- `research/` — Research notes and reuse survey
- `decisions/` — Decision records (Agora-compatible)
- `sources/` — Original source materials

## Sources

- [pymdp](https://github.com/infer-actively/pymdp)
- [Millidge Retrospective (2024)](https://www.beren.io/2024-07-27-A-Retrospective-on-Active-Inference/)
- [IWAI 2026 Workshop](https://iwaiworkshop.github.io/)
- CompuFlair video transcript (in `sources/`)

## Review Schedule

Next review: September 2026 — check for LLM + active inference implementations and IWAI 2026 proceedings.

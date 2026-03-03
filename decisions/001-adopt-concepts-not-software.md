# Decision 001: Adopt Active Inference Concepts, Not Software

**Date:** 2026-03-02
**Status:** Accepted
**Deciders:** Greg Mulvihill (executive), Orchestrate-AI (analysis)

## Context

Active inference is a theoretical framework from computational neuroscience (Friston, 2010) that unifies perception, learning, and action under variational free energy minimization. Evaluated for potential adoption into the Cogent Echo AI portfolio.

## Decision

**Adopt active inference as a design philosophy and vocabulary. Do not adopt as a software dependency.**

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| **A: Full adoption (pymdp + custom code)** | Theoretically principled, unique differentiation | Toy-grade tooling, no scalability, RL-equivalent at scale, no production examples |
| **B: Adopt concepts only (selected)** | Zero dependency risk, enriches design vocabulary, compatible with existing stack | No working active inference system to demo |
| **C: No adoption** | Zero effort, no distraction | Misses genuinely useful design concepts |
| **D: Monitor only** | Low effort | Delays potential value from applicable concepts |

## Rationale

1. **RL equivalence:** Beren Millidge (former active inference researcher) demonstrated that active inference and RL are mathematically equivalent when implemented with neural networks. The expected free energy exploration bonus has derivation issues.

2. **Ecosystem immaturity:** pymdp v0.0.7.1 handles only discrete state spaces. No production deployments exist in any domain.

3. **Concept value is real:** Markov blankets, precision weighting, generative models, and free energy as prediction error are genuinely useful design concepts for agent architecture, independent of the active inference software stack.

4. **Standard tooling is better:** Model-based RL (Dreamer, MuZero) achieves the same practical outcomes with mature, scalable tooling.

## Consequences

- Cogent Echo architects should learn the vocabulary (Markov blankets, precision, free energy)
- No pymdp or active inference library will be added to any Cogent Echo project
- Agent boundary design in Reasoning & Verification layer may adopt Markov blanket formalism
- LLM + active inference hybrid space will be monitored with September 2026 review

## Confidence

85% — High confidence that software adoption is premature. Moderate confidence that concept adoption adds value (dependent on how well patterns translate to existing architecture).

## Review Schedule

September 2026 — check for:
- New production deployments
- LLM + active inference working implementations
- IWAI 2026 workshop proceedings
- Changes in ecosystem maturity (pymdp or alternatives)

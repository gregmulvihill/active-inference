# Reuse-First Survey: Active Inference Ecosystem

**Date:** 2026-03-02
**Method:** Reuse-first cascade per `~/bootstrap/reference/reuse-first-hierarchy.md`

## Tier 1: Existing Solutions (Adopt as-is)

| Solution | Version | Language | Maturity | Scalability | Verdict |
|----------|---------|----------|----------|-------------|---------|
| **pymdp** | 0.0.7.1 | Python | JOSS published 2022, active development | Discrete state spaces only. Hand-designed matrices explode combinatorially. | Toy-grade. Not adoptable for production. |
| **SPM** | 12+ | MATLAB | Mature for neuroscience | Neuroscience-specific, MATLAB dependency | Wrong domain entirely |
| **RxInfer.jl** | Active | Julia | Growing, message-passing approach | Better scalability via message passing | Language mismatch (Julia, not Python) |

**Verdict:** No adopt-as-is solution for production active inference in Python.

## Tier 2: Existing Components (Compose)

| Component | Role | Gap to Active Inference |
|-----------|------|------------------------|
| PyTorch / JAX | Neural network backends for deep active inference | Need custom active inference layer |
| NumPyro / Pyro | Variational inference primitives | Building blocks, unassembled |
| Gymnasium | Environment interfaces for agent testing | Standard, works with anything |
| Dreamer / MuZero | Model-based RL (achieves same outcomes) | Already does what deep active inference does, with better tooling |

**Verdict:** Components exist but require significant custom assembly. Standard model-based RL tooling achieves the same practical outcomes with less effort.

## Tier 3: Existing Source (Fork/Adapt)

| Repository | Description | Assessment |
|------------|-------------|------------|
| infer-actively/pymdp | Primary Python implementation | Could fork, but discrete-state limitation is fundamental |
| Various deep active inference papers | Companion code for papers | Research code, not production quality |

**Verdict:** Nothing worth forking. The discrete-state limitation is architectural, not patchable.

## Tier 4: Snippets / Patterns

| Pattern | Source | Value |
|---------|--------|-------|
| Free energy calculation | pymdp tutorials | Educational, helps understand the math |
| Markov blanket design | Friston papers | Conceptual framework, applicable as design pattern |
| Precision weighting | Active inference literature | Translatable to attention mechanisms in standard architectures |

**Verdict:** Design patterns are the highest-value extraction from this ecosystem.

## Tier 5: Research / Prior Art

| Resource | Type | Key Finding |
|----------|------|-------------|
| Friston (2010) | Foundational paper | Free energy principle as unified brain theory |
| Parr, Pezzulo & Friston (2022) | Textbook | Comprehensive treatment of active inference |
| Millidge (2024) retrospective | Critical assessment | **Active inference ≈ RL. No special sauce.** |
| IWAI 2026 workshop | Conference | Active community, research-focused not production-focused |
| arxiv 2412.10425 | LLM + active inference | Theoretical, no code |
| npj Digital Medicine (2025) | LLM prompt reliability | Narrow but practical application |

**Verdict:** Rich theoretical literature. Production evidence is absent.

## Tier 6: Build from Scratch

**Not warranted.** Given:
1. Mathematical equivalence to RL at scale
2. No demonstrated advantage over model-based RL
3. Immature tooling ecosystem
4. Zero production deployments

Building from scratch would be reinventing model-based RL with extra Bayesian formalism and no practical payoff.

## Summary Decision

```
Reuse-First Cascade Result:
├── Tier 1 (Adopt): EXHAUSTED — nothing production-ready
├── Tier 2 (Compose): AVAILABLE — but standard RL tooling is better
├── Tier 3 (Fork): EXHAUSTED — fundamental limitations
├── Tier 4 (Patterns): ★ BEST VALUE — design patterns extractable
├── Tier 5 (Research): ★ RICH — informs architectural thinking
└── Tier 6 (Build): NOT WARRANTED
```

**Recommendation:** Extract design patterns (Tier 4) and research insights (Tier 5). Skip Tiers 1-3 and 6.

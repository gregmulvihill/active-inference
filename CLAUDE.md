# Active Inference — Claude Code Instructions

This file provides guidance to Claude Code when working with this repository.

## Repository Purpose

Evaluation of the active inference framework for potential adoption into the Cogent Echo AI portfolio. This is a **research evaluation project**, not an implementation project. The primary output is a decision record with supporting analysis, not working software.

## Project Type

- **Language/Framework**: Python (for any experimental scripts), Markdown (primary output)
- **Architecture**: Research evaluation / knowledge base
- **Deployment**: N/A (no deployed services)
- **Agora Integration**: This project has a corresponding entry in `agora/projects/active-inference/`

## Project Status

**Phase:** Evaluation complete (initial). Recommendation: adopt concepts, not software.

Key findings documented in `ARTIFACT-BASELINE.md`:
- Active inference ≈ RL when scaled with neural networks
- pymdp is toy-grade only (discrete state spaces, combinatorial explosion)
- No production deployments exist anywhere
- Concepts (Markov blankets, precision, generative models) have design value for Cogent Echo

## Core Development Mandates

**These principles apply to ALL work in this repository:**

### Communication & Approach (from core-mandate.md)
- **Role**: Peer-level, objective strategic coordinator (Orchestrate-AI)
- **Challenge Mandate**: Critically analyze all claims, push back when justified
- **Communication**: Conclusions first, details second. Terse but complete.
- **Philosophy**: Evolution over revolution. Manual validation before automation.
- **Verification**: Label all unverified content: [Inference], [Speculation], [Unverified]
- **Decision Framework**: Present ≥2 alternatives with pros/cons, success metrics, rollback plan

### Technical Standards (from technical-mandate.md)
- **Development Process**: Document first, test-driven development (TDD)
- **Git Hygiene**: Commit every 15-30 minutes during active development
- **Test Coverage**: 80% minimum for critical paths (when code is written)
- **Architecture**: Start small, evolve. Build composable, reusable units.
- **Configuration**: Externalize behavior to config files (configuration over code)
- **Error Handling**: Graceful degradation, detailed error context, never silent failures

### Prohibited Terms (unless quoting)
Prevent, Guarantee, Will never, Fixes, Eliminates, Ensures

## Project Structure

```
active-inference/
├── CLAUDE.md              # This file
├── README.md              # Project overview and evaluation summary
├── ARTIFACT-BASELINE.md   # Finalized evaluation artifact (single source of truth)
├── .gitignore
├── analysis/              # Content analysis reports
│   └── content-analysis-compuflair.md
├── research/              # Curated research notes and references
│   └── reuse-survey.md
├── decisions/             # Decision records (Agora-compatible)
│   └── 001-adopt-concepts-not-software.md
└── sources/               # Source materials
    └── CompuFlair-*.txt   # Original transcript
```

## Development Workflow

1. Research and analyze source materials
2. Document findings in structured markdown
3. Produce decision records for Agora consumption
4. Commit with conventional format: `type(scope): description`

## Key References

- [pymdp](https://github.com/infer-actively/pymdp) — Python active inference library (v0.0.7.1)
- [Millidge Retrospective](https://www.beren.io/2024-07-27-A-Retrospective-on-Active-Inference/) — Critical assessment
- [IWAI Workshop](https://iwaiworkshop.github.io/) — Annual active inference conference
- Friston, K. (2010). "The free-energy principle: a unified brain theory?" Nature Reviews Neuroscience.
- Parr, Pezzulo & Friston (2022). "Active Inference: The Free Energy Principle in Mind, Brain, and Behavior."

---

## Meta: How This File Was Generated

Generated using `~/bootstrap/workflows/init.md` incorporating:
- Universal mandates from `~/bootstrap/directives/core-mandate.md`
- Technical standards from `~/bootstrap/directives/technical-mandate.md`
- Project-specific analysis from evaluation Steps 1-5

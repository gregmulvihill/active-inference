# ADR 001: Discrete State Spaces Only

**Status:** Accepted
**Date:** 2026-03-02
**Context:** Active inference can operate over continuous or discrete state spaces. Continuous requires neural network function approximators (deep active inference). Discrete uses exact matrix operations.
**Decision:** Implement discrete state spaces only for the initial version.
**Consequences:** Math is exact and interpretable. State spaces must remain small (combinatorial explosion). Cannot handle pixel inputs or continuous control. Extension to deep active inference is a separate project.
**Alternatives Considered:** Deep active inference with VAE-based state inference — deferred to future work if discrete version proves the concepts.

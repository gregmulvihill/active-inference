# ADR 002: Empirical Prior for VFE Computation

**Status:** Accepted
**Date:** 2026-03-02
**Context:** VFE has a complexity term KL(q(s) || prior). Originally used model.D (initial state prior) for all steps. When the agent moves away from the start state, KL against the initial prior becomes large — penalizing the agent for having moved at all.
**Decision:** After the first step, use the predicted next state (from transition model B) as the empirical prior instead of the fixed D.
**Consequences:** VFE now measures surprise relative to what the agent expected given its action, not relative to the initial state. This produces more meaningful free energy signals and more stable precision adaptation.
**Alternatives Considered:** Fixed small prior weight — partially addressed the issue but didn't eliminate it.

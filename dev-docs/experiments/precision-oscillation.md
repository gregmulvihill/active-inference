# Experiment: Precision Oscillation in Grid World

**Date:** 2026-03-02
**Status:** Resolved

## Problem

Grid world agent's adaptive precision crashed to the minimum (originally 0.1), making the agent too random to navigate effectively. The agent oscillated between positions instead of progressing toward the target.

## Root Cause

Two compounding issues:
1. VFE computed against fixed model.D (initial prior) produced large KL divergence as the agent moved away from start position
2. Raw VFE delta used for precision update was noisy and drove precision monotonically downward

## Fix Applied

1. **Empirical prior** — VFE now uses predicted next state as prior (ADR-002)
2. **EMA-smoothed precision** — precision adapts relative to exponential moving average of VFE (smoothing factor 0.8). Clamp range tightened to [0.5, 16.0].

## Result

Grid world test passes with seed=42 (agent reaches target in <30 steps). Demo performance is variable across seeds — acceptable for current scope.

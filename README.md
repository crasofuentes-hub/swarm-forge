# Swarm Forge v1.0

[![CI](https://github.com/crasofuentes-hub/swarm-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/crasofuentes-hub/swarm-forge/actions/workflows/ci.yml)

**Swarm Forge** is an autonomous training optimizer for nanoGPT-style models.

It improves training through **governed multi-agent patch search**, **empirical validation**, and **rollback-safe execution**.

## Current Best Measured Result

Swarm Forge has reduced TinyShakespeare validation loss from approximately **4.1736** to **1.6318** under autonomous multi-agent search.

See [RESULTS.md](./RESULTS.md) for the current evidence summary.

---

## Core Thesis

Most training repositories optimize a model.

Swarm Forge optimizes the **optimizer**.

Instead of relying only on a fixed training script and manual tuning, Swarm Forge runs a governed loop in which specialized agents:

- propose training patches,
- score candidate changes,
- vote under explicit thresholds,
- apply only accepted modifications,
- preserve rollback safety and checkpoints.

This makes the repository interesting not only as a language-model experiment, but as a framework for **autonomous training improvement**.

---

## What This Project Is

Swarm Forge is:

- a **multi-agent training optimization system**,
- a **governed patch search engine** for model improvement,
- a **rollback-safe runtime** for controlled self-modification,
- a **checkpointed experimentation framework**,
- an **AutoML-style research system** built around measurable training progress.

---

## What This Project Is Not

Swarm Forge is **not**:

- a claim of AGI,
- a foundation model,
- a proof of general self-improving intelligence,
- a production-scale LLM training stack,
- a benchmark winner across large public leaderboards.

It is a **research-grade engineering system** focused on autonomous optimization of training dynamics.

---

## Why It Is Interesting

Most small-model experiments stop at:

- train script,
- hyperparameter sweep,
- one-off manual tuning,
- static results.

Swarm Forge goes further by combining:

- multi-agent proposal generation,
- threshold-based governance,
- empirical scoring,
- transactional patch application,
- rollback and persistence,
- resumed improvement from checkpoints.

That combination is much rarer than training another small GPT.

---

## System Shape

Swarm Forge defines **120 agents** across **12 roles**.

Each living agent proposes exactly one patch per cycle.

Patches are evaluated under weighted governance:

- Evaluators collectively control **50%** of the vote.
- Other agents collectively control **50%** of the vote.

Only patches that clear approval and score thresholds are applied.

---

## Current Evidence

At the time of this update, the project has already demonstrated:

- reproducible baseline measurement,
- autonomous reduction of TinyShakespeare validation loss,
- resumed continuation from checkpoints,
- accepted winning patches during training,
- CI-backed verification,
- rollback-safe patch application,
- package-based Python distribution with CLI entrypoints.

Current empirical trajectory:

- baseline: **~4.1736**
- focused GPU search v2: **1.8794**
- resumed v3 improvement: **1.6318**

---

## Why TinyShakespeare Still Matters

TinyShakespeare is small, cheap, reproducible, and useful for exposing:

- convergence behavior,
- patch acceptance quality,
- training loop instability,
- optimization noise,
- governance flaws,
- checkpoint correctness.

It is valuable because it is a sharp proving ground for training optimization systems.

---

## Engineering Commitments

Swarm Forge prioritizes:

1. **Measurable improvement over theatrical complexity**
2. **Rollback safety over reckless mutation**
3. **Governed change over arbitrary self-modification**
4. **Checkpoint continuity over ephemeral runs**
5. **Empirical validation over narrative claims**

---

## Current Limits

This project does **not yet** prove:

- validation loss below 1.4 on TinyShakespeare,
- broad transfer to other datasets,
- stable superiority across architectures,
- large-scale distributed training robustness.

Those are next-stage goals.

The current achievement is narrower and important in its own right:

> autonomous, governed, measurable improvement of model training.

---

## What GitHub Should Tell the World

Swarm Forge is not "another tiny GPT repo."

It is a **training optimization system**.

Its significance is not just that it trains a model.
Its significance is that it organizes autonomous training improvement under evidence, thresholds, rollback, and checkpoints.

That is the real idea.

---

## Status

**Version:** v1.0  
**State:** experimental, test-backed, CI-validated, improving  
**Current best TinyShakespeare val_loss:** **1.6318**  
**Near-term target:** **< 1.4 on a single GPU**

---

## Next Milestones

- push TinyShakespeare validation loss below **1.7**
- push below **1.5**
- reach or beat **1.4**
- improve patch selection so accepted patches influence convergence more strongly
- strengthen result reporting and reproducibility artifacts
- generalize beyond TinyShakespeare

---

## Final Principle

Swarm Forge should be judged less by how many agents it has and more by whether it can keep improving training under controlled, auditable, checkpointed search.
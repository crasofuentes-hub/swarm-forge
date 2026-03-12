# Swarm Forge Roadmap

## Phase 1 — Proof of Autonomous Improvement (current)

Goals:

- measurable improvement of TinyShakespeare validation loss
- stable checkpoint resume
- governance thresholds for patch application
- rollback-safe mutation
- reproducible runs

Current best result:

TinyShakespeare validation loss reduced from **~4.17 -> 1.8454**

---

## Phase 2 — Stronger Training Optimization

Targets:

- push TinyShakespeare below **1.7**
- reach **1.5**
- approach **1.4**

Engineering goals:

- better patch selection heuristics
- improved evaluator scoring
- faster convergence cycles
- more consistent patch winners

---

## Phase 3 — Cross Dataset Validation

Test Swarm Forge against:

- TinyStories
- OpenWebText subsets
- WikiText small splits

Goal:

demonstrate that autonomous optimization is not dataset-specific.

---

## Phase 4 — Architecture Experiments

Allow agents to modify:

- optimizer configuration
- schedule policy
- attention architecture
- layer normalization strategies

Goal:

move from hyperparameter tuning toward architecture discovery.

---

## Phase 5 — Distributed Swarm

Introduce distributed agent workers via Ray or similar frameworks.

Goals:

- parallel patch exploration
- larger search spaces
- faster convergence cycles

---

## Long Term Vision

A reusable system capable of improving training loops under governed search and empirical validation.
# Ablation Plan

## Purpose

Swarm Forge should be evaluated not only by final validation loss, but by whether its governance and agent structure add measurable value.

## Minimum Ablation Set

### A. Static baseline
No autonomous patch application.

### B. Agents generate patches, but no patches are applied
Measure whether agent evaluation alone provides useful signals.

### C. Full governed swarm
Current default governed configuration.

### D. Reduced-role swarm
Keep only the highest-signal roles:

- HyperparamTuner
- SpeedDemon
- MemoryWarden
- Evaluator
- Arbiter

## Metrics To Compare

For each ablation:

- best val_loss
- train_loss
- perplexity
- throughput_tokens_per_sec
- number of accepted patches
- number of rejected patches
- time per cycle

## Why This Matters

Without ablations, Swarm Forge is interesting.

With ablations, Swarm Forge becomes scientifically inspectable.
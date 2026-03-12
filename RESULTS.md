# Swarm Forge Results

## Current Best Empirical Result

Swarm Forge has reduced TinyShakespeare validation loss from approximately **4.1736** to **1.8454** under governed multi-agent training search.

## Run Summary

| Run | Mode | Configuration | Best Validation Loss | Notes |
|---|---|---|---:|---|
| Baseline probe | CPU baseline_only | 2 layers / 2 heads / 64 embd | 4.1736 | Reproducible baseline measurement |
| GPU focused search v2 | Autonomous search | 8 layers / 8 heads / 512 embd | 1.8794 | Best checkpoint at cycle 7 |
| GPU focused search v3 | Resumed autonomous search | 8 layers / 8 heads / 512 embd | 1.8454 | Improved after checkpoint resume |

## Why This Matters

Most small-model training repositories optimize a model through static code and manual tuning.

Swarm Forge instead treats training improvement as a governed search problem:

- multiple specialized agents propose changes,
- evaluators score empirical impact,
- arbitration applies only threshold-clearing patches,
- rollback-safe execution prevents unsafe mutation,
- checkpoints preserve progress across runs.

This repository is therefore not just a model trainer.

It is a **training optimizer**.

## Current Interpretation

The project has not yet demonstrated validation loss below 1.4 on TinyShakespeare.

However, it has already demonstrated:

- stable autonomous improvement,
- checkpointed continuation,
- patch application under governance,
- measurable reduction in validation loss,
- a reusable optimization framework rather than a single static training script.

## Next Target

**Near-term target:** push TinyShakespeare validation loss from **1.8454** toward **< 1.4** on a single GPU through continued autonomous search and tighter patch acceptance logic.
# Reproducibility

## Goal

This document explains how to verify the current best known TinyShakespeare result for Swarm Forge.

## Dataset

Expected dataset location:

```text
data\tinyshakespeare
```

## Best Known Checkpoint

```text
runs\gpu_focused_search_v3\checkpoints\checkpoint_best_cycle_0007_valloss_1.6318.pt
```

## Reference Training Family

```powershell
python -m swarm_forge `
  --focused-search `
  --reduced-roles-mode `
  --device cuda `
  --amp `
  --dtype float16 `
  --batch-size 12 `
  --block-size 256 `
  --n-layer 8 `
  --n-head 8 `
  --n-embd 512 `
  --learning-rate 2e-4 `
  --dropout 0.2 `
  --eval-iters 30 `
  --max-iters-per-cycle 300 `
  --patch-trial-train-steps 50 `
  --max-cycles 10 `
  --cycle-seconds 900 `
  --output-dir runs\gpu_focused_search_v3 `
  --data-dir data\tinyshakespeare
```

## What To Compare

Important outputs:

- train_loss
- val_loss
- perplexity
- throughput_tokens_per_sec
- accepted patches
- saved checkpoints

## Evidence Files

```text
runs\gpu_focused_search_v3\logs\metrics.jsonl
runs\gpu_focused_search_v3\logs\swarm.log
runs\gpu_focused_search_v3\checkpoints\
```

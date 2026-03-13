# Experiment Registry

This file records the major experiments performed with Swarm Forge.

## TinyShakespeare

### Baseline Probe

- Dataset: TinyShakespeare
- Model: 2 layers / 2 heads / 64 embedding
- Device: CPU
- Validation Loss: ~4.1736

### GPU Focused Search v2

- Dataset: TinyShakespeare
- Model: 8 layers / 8 heads / 512 embedding
- Device: GTX 1660 Ti
- Best Validation Loss: ~1.8794
- Checkpoint: checkpoint_best_cycle_0007_valloss_1.8794.pt

### GPU Focused Search v3

- Resume from v2 checkpoint
- Best Validation Loss: ~1.6318
- Checkpoint: checkpoint_best_cycle_0007_valloss_1.6318.pt

Trajectory:

```text
4.1736 -> 1.8794 -> 1.8454 -> 1.7546 -> 1.7031 -> 1.6318
```

### GPU Focused Search v4

- Resume from v3 checkpoint
- Result: regression (~1.83)
- Conclusion: configuration degraded convergence

## Notes

- All runs executed on a single GPU unless stated otherwise.
- Logs available under runs/ directories.
- Metrics recorded in metrics.jsonl.

Future registry entries should follow this structure.

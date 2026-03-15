# TinyShakespeare Status

## Current status
TinyShakespeare is now under a formal tracked regime in Swarm Forge.

## GitHub milestones already published
- Freeze tag: `freeze-guarded-1.5932`
- Runtime recovery fix: `bc8b160`
- Baselines/ablations doc: `09f79f7`
- Canonical reproduction script: `ec756eb`

## Best historical checkpoint marker
- `runs\guarded_real_from_16094\checkpoints\checkpoint_best_cycle_0001_valloss_1.5932.pt`

## Reproducible baseline under current runtime
- **1.6043**

## Key findings
1. Monolithic continuation from this regime is unstable.
2. Guarded continuation is essential.
3. The runtime-reproducible baseline is currently `1.6043`.
4. The checkpoint filename `1.5932` remains part of provenance, but not the currently reproduced baseline value.

## Formal evidence already recorded
- Baseline and ablation results:
  - `TINYSHKESPEARE_BASELINES.md`
- Experiment registry:
  - `EXPERIMENT_REGISTRY.md`
- Ablation protocol:
  - `ABLATION_PROTOCOL_TINYSHAKESPEARE.md`
- Canonical reproduction script:
  - `scripts\reproduce-tinyshakespeare-reference.ps1`

## Current conclusion
TinyShakespeare is sufficiently stabilized and documented to move to the next roadmap step: evaluation on a second dataset.
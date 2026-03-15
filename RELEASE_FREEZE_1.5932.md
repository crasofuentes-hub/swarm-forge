# Swarm Forge Freeze 1.5932

Stable fine-tuning freeze validated on machine **MARLINE**.

## Summary
This freeze locks the current stable training regime and preserves the best validated result achieved with guarded fine-tuning.

## Validated result
- Commit: `786c762`
- Tag: `freeze-guarded-1.5932`
- Best validated `val_loss`: **1.5932**
- Best validated checkpoint:
  `runs/guarded_real_from_16094/checkpoints/checkpoint_best_cycle_0001_valloss_1.5932.pt`

## What changed
- Hard block for `model_arch` in fine-tuning regime.
- Resume path now enforces the CLI learning rate after optimizer state restoration.
- Guarded training by chunks with intermediate validation.
- Rollback to the best state found inside the cycle.

## Why this freeze matters
Previous long-cycle runs could improve temporarily and then degrade before the cycle ended. This freeze preserves the best point reached inside the cycle instead of trusting the final state blindly.

## Validated trajectory
- `1.6208`
- `1.6130`
- `1.6094`
- `1.5932`
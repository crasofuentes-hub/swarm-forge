# Extensive Campaign Plan

- campaign_id: `camp-extensive-grid-20260319`
- created_at: `2026-03-19T15:53:10.356392+00:00`
- dataset: `tinyshakespeare`
- objective_metric: `val_loss`
- maximize: `False`
- device: `cuda`
- dtype: `bfloat16`
- amp_enabled: `True`
- batch_size: `8`
- micro_batch_size: `8`
- block_size: `32`
- eval_iters: `8`
- total_trials: `540`

## Search Space

- learning_rate: `[0.0003, 0.0004, 0.00045, 0.0005, 0.0006]`
- weight_decay: `[0.0, 0.01, 0.02, 0.05]`
- dropout: `[0.05, 0.1, 0.15]`
- label_smoothing: `[0.0, 0.01, 0.02]`
- patch_trial_train_steps: `[100, 1000, 8000]`

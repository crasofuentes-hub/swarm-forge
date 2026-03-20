# OpenWebText Real Smoke Report

## Objective
Validate that Swarm Forge can train end-to-end on a real web corpus using nanoGPT vanilla assumptions.

## Dataset
- dataset: `openwebtext`
- tokenizer: `gpt2`
- vocab_size: `50257`

## Training Config
- device: `cuda`
- dtype: `bfloat16`
- amp_enabled: `true`
- batch_size: `8`
- micro_batch_size: `8`
- block_size: `128`
- n_layer: `4`
- n_head: `4`
- n_embd: `256`
- dropout: `0.1`
- learning_rate: `3e-4`
- weight_decay: `0.0`
- label_smoothing: `0.0`
- train_steps: `20`

## Baseline Eval
- train_loss: `10.853143692016602`
- val_loss: `10.854913711547852`
- perplexity: `51787.99846548604`

## After 20 Steps
- train_loss: `9.721878051757812`
- val_loss: `9.743942260742188`
- perplexity: `17050.627076951703`

## Deltas
- delta_train_loss: `-1.131265640258789`
- delta_val_loss: `-1.110971450805664`

## Conclusion
OpenWebText is now functioning as a real corpus benchmark inside Swarm Forge.
TinyShakespeare should remain only as a toy/smoke benchmark, not the main training benchmark.
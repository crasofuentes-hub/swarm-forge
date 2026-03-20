# OpenWebText Baseline 200 Report

## Objective
Establish the first reproducible real baseline over OpenWebText inside Swarm Forge.

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
- train_steps: `200`

## Baseline Eval
- train_loss: `10.857220649719238`
- val_loss: `10.856060981750488`
- perplexity: `51847.44738844794`

## After 200 Steps
- train_loss: `7.266928195953369`
- val_loss: `7.127401828765869`
- perplexity: `1245.6363823226473`

## Train Stats
- global_step: `200`
- train_loss_recent: `7.813573150634766`
- train_throughput_tokens_per_sec: `20477.263827912397`

## Deltas
- delta_train_loss: `-3.590292453765869`
- delta_val_loss: `-3.728659152984619`

## Checkpoint
- path: `runs/openwebtext_baseline_200/checkpoints/checkpoint_baseline_200.pt`

## Conclusion
Swarm Forge now has a real, reproducible OpenWebText baseline with checkpoint generation on GPU.
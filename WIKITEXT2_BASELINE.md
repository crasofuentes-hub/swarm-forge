# WikiText-2 Baseline

## Objective
First formal second-dataset baseline for Swarm Forge.

## Machine of record
- Machine: `MARLINE`
- Repo: `C:\repos\swarm-forge`
- Python: `C:\repos\swarm-forge\.venv\Scripts\python.exe`

## Dataset
- Name: `wikitext2`
- Variant: raw split files
- Files:
  - `data\wikitext2\wiki.train.raw`
  - `data\wikitext2\wiki.valid.raw`
  - `data\wikitext2\wiki.test.raw`

## Run configuration
- device = cuda
- amp = on
- dtype = float16
- batch_size = 8
- block_size = 256
- n_layer = 8
- n_head = 8
- n_embd = 512
- dropout = 0.2
- learning_rate = 1e-4
- max_iters_per_cycle = 40
- eval_iters = 20
- max_cycles = 1
- cycle_seconds = 900
- focused_search = true
- reduced_roles_mode = true
- patch_trial_train_steps = 20

## Key observed lines
- baseline validation loss: `7.0508`
- chunk 20 validation loss: `3.5703`
- chunk 40 validation loss: `3.0648`
- saved improved checkpoint:
  - `runs\wikitext2_baseline\checkpoints\checkpoint_best_cycle_0001_valloss_3.0648.pt`
- `runs\wikitext2_exp1_lr5e5\checkpoints\checkpoint_best_cycle_0001_valloss_2.6720.pt`
- `runs\wikitext2_exp4_lr2p5e5\checkpoints\checkpoint_best_cycle_0001_valloss_2.5970.pt`

## Final summary
- train_loss = `3.3072`
- val_loss = `3.0648`
- delta = `-3.9943`
- perplexity = `21.4303`
- bleu_like = `2.0833`
- throughput_tokens_per_sec = `9864.2929`
- alive = `120`
- winners = `0`

## Conclusion
Swarm Forge successfully completed a controlled second-dataset baseline on WikiText-2 and improved validation loss strongly from `7.0508` to `3.0648` within one guarded cycle.

## Scientific interpretation
This is sufficient evidence that the runtime is no longer restricted to TinyShakespeare and that guarded training remains operational on a second dataset.
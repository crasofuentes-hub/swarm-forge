# TinyShakespeare Baselines and Reproducible Results

## Scope
Formal baseline/ablation record for the current Swarm Forge TinyShakespeare regime.

## Machine of record
- Machine: `MARLINE`
- Repo: `C:\repos\swarm-forge`
- Python: `C:\repos\swarm-forge\.venv\Scripts\python.exe`

## Runtime state
- Current main commit that restored valid runtime: `bc8b160`
- Freeze tag previously established: `freeze-guarded-1.5932`

## Reference checkpoint under study
- `runs\guarded_real_from_16094\checkpoints\checkpoint_best_cycle_0001_valloss_1.5932.pt`

## Important note
Although the checkpoint filename includes `1.5932`, the current runtime reproduces it at a **baseline validation loss of 1.6043**.  
For scientific comparison, the reproducible baseline is the runtime-observed value, not only the checkpoint filename.

---

## Baseline / Ablation Table

| ID | Regime | Resume checkpoint | LR | Guarded | Result |
|---|---|---|---:|---|---:|
| B0 | Historical best filename marker | `checkpoint_best_cycle_0001_valloss_1.5932.pt` | 1e-4 | yes | 1.5932 (filename / historical marker) |
| B1 | Reproduced guarded reference | `checkpoint_best_cycle_0001_valloss_1.5932.pt` | 1e-4 | yes | **1.6043** |
| A1 | Monolithic continuation 60 | same | 1e-4 | no | **2.1326** |
| A2 | Guarded continuation 60 | same | 1e-4 | yes | **1.6043** |
| A3 | Guarded continuation 60 | same | 5e-5 | yes | **1.6043** |

---

## Interpretation

### Reproduced guarded reference
- Effective optimizer LR after resume: `0.00010000`
- Reproduced baseline validation loss: `1.6043`
- First guarded chunk immediately degraded to `1.6995`
- Guarded stop triggered correctly
- Final preserved result: `1.6043`

### Monolithic continuation
- Starting from the same regime, monolithic continuation degraded strongly
- Observed result: `2.1326`
- Conclusion: monolithic continuation is unstable and not acceptable in this regime

### Guarded continuation
- Guarded controller prevented destructive continuation
- However, no further improvement was obtained from the reproduced `1.6043` regime under the tested settings
- LR `1e-4` and LR `5e-5` both failed to beat the reproduced baseline

---

## Current scientific conclusion
1. The guarded controller is essential.
2. Monolithic continuation is formally rejected for this regime.
3. The runtime-reproducible baseline currently defended by evidence is `1.6043`.
4. The historical `1.5932` checkpoint name remains important as provenance, but not as the current reproduced baseline value.

## Next formal step
Run the next ablation only after logging a single explicit hypothesis and a single changed variable.
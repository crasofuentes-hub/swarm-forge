# Dataset Generalization Blocker

## Status
Swarm Forge cannot yet execute a real second-dataset experiment.

## Evidence
Current runtime is still hard-coupled to TinyShakespeare:

- Only dataset class currently implemented:
  - `class TinyShakespeareData`
- Engine initialization hard-codes:
  - `self.dataset = TinyShakespeareData(swarm_cfg.data_dir, train_cfg, self.logger)`
- Dataset implementation assumes:
  - `input.txt`
  - TinyShakespeare download URL
  - TinyShakespeare-specific data path defaults

## Scientific consequence
The roadmap item "second dataset" is not experimentally open yet.
At this point, any claim of multi-dataset generalization would be premature.

## Formal blocker
Before evaluating WikiText-2 or any other dataset, Swarm Forge needs a dataset abstraction layer.

## Required next engineering step
Introduce a dataset interface/factory so that the runtime can instantiate datasets by name instead of directly binding to `TinyShakespeareData`.

## Minimal acceptance criterion for unblocking
- dataset selection is explicit
- engine no longer hard-codes `TinyShakespeareData`
- at least one second dataset can be loaded without modifying core runtime logic
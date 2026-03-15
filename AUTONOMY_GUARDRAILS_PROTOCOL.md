# Autonomy Guardrails Protocol

## Objective
Introduce experimental autonomy without allowing unconstrained runtime drift.

## Current project status
Swarm Forge has now:
- a closed TinyShakespeare regime
- documented baselines and ablations
- a canonical reproduction script
- a working second-dataset baseline on WikiText-2

## Scope of allowed autonomy
At this stage, agents may only propose and select bounded experiment configurations.
They may not yet:
- rewrite arbitrary training code
- change dataset loader implementations
- modify checkpoint semantics
- bypass rollback or guarded stop logic

## First autonomy phase
Autonomy is restricted to proposing experiment metadata only:
- hypothesis
- single changed variable
- acceptance criterion
- rollback condition

## Allowed autonomous variables
- learning_rate
- max_iters_per_cycle
- batch_size
- guarded chunk size
- evaluation budget

## Forbidden autonomous variables
- architecture topology
- checkpoint serialization format
- dataset parser logic
- signal handling
- runtime guard disabling
- arbitrary source-code mutation

## Acceptance rule
An autonomous proposal is valid only if:
1. it changes one variable
2. it names one hypothesis
3. it defines one success criterion
4. it preserves rollback and guarded execution semantics

## Immediate next engineering target
Represent autonomous experiment proposals as structured records first.
Do not let agents directly edit runtime code yet.
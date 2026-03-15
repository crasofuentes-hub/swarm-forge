# Second Dataset Protocol

## Selected dataset
- WikiText-2

## Why WikiText-2
1. It is a standard language-model benchmark.
2. It is meaningfully different from TinyShakespeare.
3. It is still feasible on the current single-GPU setup.
4. It is appropriate for testing whether Swarm Forge generalizes beyond a tiny stylized corpus.

## Scientific objective
Test whether the guarded training regime and the current Swarm Forge control logic remain stable on a second dataset.

## Formal rule
Before running any experiment:
- keep one dataset only
- keep one hypothesis only
- keep one changed variable only

## First experiment on the second dataset
### Hypothesis
The current guarded training regime is stable enough to complete one controlled run on WikiText-2 without destructive drift.

### Acceptance criterion
- run completes successfully
- no structural collapse
- no catastrophic validation explosion
- final metrics and logs are recorded reproducibly

## Constraints
- Do not modify architecture first.
- Do not open multiple tuning dimensions.
- Do not add autonomy changes before baseline execution on WikiText-2.
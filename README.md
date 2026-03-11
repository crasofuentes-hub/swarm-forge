# Swarm Forge v1.0

**Swarm Forge** is a competitive engineering system for disciplined experimentation on a nanoGPT-style language model using TinyShakespeare as a bounded testbed.

It is built around a simple premise:

> before scaling capability, harden correctness, observability, rollback safety, and decision discipline.

This repository is not a theatrical “autonomous AI swarm.”
It is an attempt to impose engineering constraints on multi-agent model iteration.

---

## Engineering Position

Most AI agent systems fail in predictable ways:

- they optimize for spectacle over repeatability,
- they add features faster than they harden contracts,
- they confuse parallel proposal generation with real governance,
- they produce output without preserving enough evidence to justify decisions,
- they make rollback, audit, and lifecycle management an afterthought.

Swarm Forge exists to push in the opposite direction.

This project treats **correction**, **transactionality**, **evaluation discipline**, and **auditability** as primary system properties.

Not because they are glamorous.
Because they are the parts that break first in production.

---

## What This Project Is

Swarm Forge is:

- a **competitive patch-generation swarm** with explicit agent roles,
- a **single-model iterative optimization loop** over a bounded training target,
- a **transactional patch application system** with rollback,
- a **weighted voting system** with explicit thresholds,
- a **cycle-based orchestration engine** with persistent metrics and checkpoints,
- a **controlled environment** for studying whether agent diversity improves training decisions under governance.

At its core, the system is an engineering harness for structured experimentation.

---

## What This Project Is Not

Swarm Forge is **not**:

- a general AGI system,
- a self-improving intelligence in the strong sense,
- a production LLM training stack,
- a replacement for rigorous ML research methodology,
- a guarantee of reaching arbitrary target losses,
- a proof that “more agents” automatically implies better optimization,
- a fully secure distributed system,
- a benchmark winner.

It does not claim more than it can presently support.

The current implementation is a bounded research-and-engineering scaffold.
Its value lies in the clarity of its contracts and the honesty of its constraints.

---

## Why TinyShakespeare

TinyShakespeare is not used because it is impressive.

It is used because it is small, legible, reproducible, cheap to iterate on, and good enough to expose systemic failures in:

- orchestration,
- patch validation,
- rollback handling,
- evaluation drift,
- lifecycle errors,
- configuration instability,
- and governance logic.

A small dataset is an advantage when the real target is **system behavior under controlled pressure**.

---

## Core Design Commitments

### 1. Governance before expansion

The system does not apply patches just because they exist.

Every patch must survive:

- structured generation,
- validation,
- multi-agent scoring,
- weighted voting,
- threshold gating,
- conflict resolution,
- transactional application.

This is slower than naive mutation.
It is also more defensible.

### 2. Auditability before feature count

A system that can do many things but cannot explain its own state transitions is fragile.

Swarm Forge prioritizes:

- persistent metrics,
- explicit cycle logs,
- patch provenance,
- decision records,
- model checkpoints,
- failure counters,
- lifecycle events.

This makes the system easier to inspect, not just easier to demo.

### 3. Bounded experimentation

Unbounded autonomy is usually a euphemism for weak control.

Swarm Forge uses:

- fixed role counts,
- explicit cycle windows,
- exact patch quotas per agent,
- deterministic thresholds,
- constrained patch families,
- bounded failure handling.

The goal is not to simulate freedom.
The goal is to create pressure inside a governable envelope.

### 4. Honest separation of concerns

Proposal generation, scoring, arbitration, training, and persistence are separate concerns.

That separation matters because systems become dangerous and unmaintainable when generation and approval collapse into the same mechanism.

---

## System Shape

Swarm Forge defines 12 explicit roles with 10 agents per role, for a total of 120 agents:

- BugHunter
- HyperparamTuner
- LayerArchitect
- TokenizerOptimizer
- DataAugmentor
- LossEngineer
- Saboteur
- ResilienceChecker
- SpeedDemon
- MemoryWarden
- Evaluator
- Arbiter

Each living agent proposes exactly one patch per cycle.

Each patch is voted on under weighted governance:

- Evaluators collectively control **50%** of the vote.
- All other living agents collectively control **50%** of the vote.

A patch is applied only if it satisfies both:

- **approval >= 60%**
- **weighted score >= 70/100**

This is intentionally conservative.

The system prefers rejecting uncertain change over accepting noisy change.

---

## Failure Model

Swarm Forge assumes agents can fail.

Failure is not exceptional behavior.
It is part of the design space.

Each agent tracks consecutive failures.
After 3 consecutive failed cycles, the agent is declared dead.

This is not punitive logic.
It is a minimal operational hygiene rule.

Lifecycle events are first-class:

- failure accumulation,
- death,
- reassignment,
- patch rejection,
- patch application rollback.

Systems that do not model degradation explicitly are usually lying about stability.

---

## Design Philosophy

Swarm Forge is based on a narrow but durable engineering philosophy:

- **A smaller truthful system is worth more than a larger theatrical one.**
- **Reproducible behavior matters more than clever wording.**
- **Rollback capacity matters more than mutation speed.**
- **Correctness pressure matters more than novelty pressure.**
- **A hardened contract is more valuable than an impressive promise.**

This repository is therefore intentionally opinionated.

It would rather be called conservative than incoherent.

---

## Current State

This repository currently provides:

- a complete single-file implementation of the Swarm Forge runtime,
- a patchable nanoGPT-style model,
- TinyShakespeare dataset bootstrap,
- weighted voting and arbitration logic,
- persistent JSONL and CSV metrics,
- model checkpointing,
- agent lifecycle tracking,
- Ray-first execution with local fallback behavior.

This means the repository is already usable for bounded experimentation.

It does **not** mean the architecture is complete.

---

## Known Limits

Current limitations include, but are not limited to:

- single-file implementation rather than a hardened installable package,
- simplified tokenizer evolution,
- simplified BLEU-like metric rather than a rigorous sequence evaluation stack,
- no formal patch-schema versioning,
- no cross-run replay verifier,
- no deterministic artifact manifest,
- no signed decision log,
- no distributed storage or fault-tolerant execution substrate,
- no formal conflict graph beyond current patch grouping logic,
- no full benchmark suite across multiple corpora or model sizes.

These are not hidden.
They are the next engineering obligations.

---

## Roadmap

The roadmap is intentionally ordered by **integrity**, not by hype.

### Phase 1 — Contract Hardening
Priority: highest

- Split the single-file implementation into a real package layout.
- Introduce strict patch schemas and validation contracts.
- Version all decision, metrics, and checkpoint formats.
- Harden rollback guarantees with pre/post state assertions.
- Add deterministic run manifests.
- Add replayable cycle logs.
- Add compatibility checks for architecture mutations.
- Add test coverage for arbitration, thresholds, conflicts, and lifecycle transitions.

### Phase 2 — Auditability and Reproducibility
Priority: very high

- Add signed manifests and stronger provenance chains.
- Add replay mode for patch application and decision verification.
- Add deterministic seed trace outputs.
- Add structured experiment registry.
- Add checkpoint lineage metadata.
- Add stronger metric provenance and comparison baselines.

### Phase 3 — Execution Robustness
Priority: high

- Improve Ray runtime supervision.
- Add worker health heartbeats.
- Add bounded retry policies.
- Add timeout-aware orchestration.
- Add resource budgeting per agent role.
- Add stronger CPU fallback parity.
- Add better interruption recovery.

### Phase 4 — Capability Expansion
Priority: secondary

Only after the above is hardened:

- improve tokenizer strategies,
- improve architecture mutation realism,
- improve resilience testing,
- improve evaluation breadth,
- support larger corpora,
- support larger nanoGPT variants,
- add richer patch conflict semantics.

Capabilities are expansion work.
They are not foundation work.

### Phase 5 — Research Extensions
Priority: optional

- multi-objective arbitration,
- patch portfolio selection,
- richer adversarial roles,
- role-adaptive weighting,
- meta-evaluation of agent usefulness,
- cross-run policy learning.

These are interesting.
They are not urgent.

---

## Non-Goals

This project does not optimize for:

- maximal agent theatrics,
- broad claims about autonomous intelligence,
- inflated benchmark marketing,
- pseudo-scientific language about emergence,
- fragile demos designed only to look advanced.

The ambition here is narrower and more serious:

**build a system whose internal decisions can be inspected, criticized, improved, and trusted more over time.**

---

## Why This Matters

Many systems can generate change.

Far fewer can justify change.

That distinction becomes decisive the moment you care about:

- reproducibility,
- enterprise reliability,
- regression control,
- safe iteration,
- governance,
- and post-failure diagnosis.

Swarm Forge is an attempt to treat those concerns as the starting point, not the cleanup phase.

---

## Status

**Version:** v1.0  
**State:** experimental, bounded, inspectable  
**Primary value:** engineering discipline under competitive multi-agent mutation  
**Primary risk:** complexity outrunning hardening if roadmap order is ignored

---

## Final Principle

If this project succeeds, it will not be because it looked intelligent.

It will be because it became difficult to fool, difficult to drift, and easier to trust.

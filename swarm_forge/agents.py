"""Agent policy layer for Swarm Forge."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict

from .core import stable_hash, utc_now
from .patches import Patch, PatchVote


@dataclass
class AgentState:
    agent_id: str
    role: str
    alive: bool = True
    consecutive_failures: int = 0
    accepted_patches: int = 0
    rejected_patches: int = 0
    total_patches: int = 0
    total_votes_cast: int = 0
    total_runtime_sec: float = 0.0
    deaths: int = 0
    reassignments: int = 0
    last_error: str = ""
    capabilities: str = ""


class Agent(ABC):
    def __init__(self, agent_id: str, role: str, seed: int):
        self.agent_id = agent_id
        self.role = role
        self.seed = seed
        self.local_rng = random.Random(seed)
        self.state = AgentState(agent_id=agent_id, role=role, capabilities=self.describe_capabilities())

    @abstractmethod
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        raise NotImplementedError

    @abstractmethod
    def describe_capabilities(self) -> str:
        raise NotImplementedError

    def score_patch(self, patch: Patch, runtime_metrics: Dict[str, Any], runtime_state: Dict[str, Any]) -> PatchVote:
        approve = True
        score = 60.0
        reason = "Default neutral vote."
        val_loss = float(runtime_metrics.get("val_loss", 99.0))
        throughput = float(runtime_metrics.get("throughput_tokens_per_sec", 0.0))
        if patch.patch_type in {"hyperparam", "loss", "memory", "speed"}:
            score += 8.0
        if patch.patch_type == "sabotage":
            score -= 15.0
            approve = val_loss < 3.0
            reason = "Sabotage accepted only under controlled regime."
        if patch.patch_type == "model_arch":
            n_layer = int(runtime_state["model_config"]["n_layer"])
            if n_layer >= 12 and patch.payload.get("n_layer_delta", 0) > 0:
                score -= 10.0
                approve = False
                reason = "Architecture already near upper complexity bound."
        if throughput < 1000 and patch.patch_type in {"speed", "memory"}:
            score += 10.0
        score += self.local_rng.uniform(-5.0, 5.0)
        score = max(0.0, min(100.0, score))
        approve = approve and score >= 55.0
        return PatchVote(
            patch_id=patch.id,
            voter_id=self.agent_id,
            voter_role=self.role,
            approve=approve,
            score=score,
            reason=reason,
        )


class BugHunterAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        num_workers = int(runtime_state["training_config"]["num_workers"])
        payload = {
            "pin_memory": True,
            "num_workers": max(0, min(8, num_workers + self.local_rng.choice([-1, 0, 1]))),
            "persistent_workers": bool(num_workers > 0),
            "prefetch_factor": self.local_rng.choice([2, 4]),
        }
        return build_patch(self, cycle_index, "bugfix", payload, "Tune loader flags to reduce stalls and unstable worker churn.")

    def describe_capabilities(self) -> str:
        return "Detects training loop inefficiencies, unstable worker settings, and latent dead-config paths."


class HyperparamTunerAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        tcfg = runtime_state["training_config"]
        lr = float(tcfg["learning_rate"]) * self.local_rng.choice([0.85, 0.92, 1.05, 1.12])
        wd = float(tcfg["weight_decay"]) * self.local_rng.choice([0.8, 0.95, 1.05])
        batch_size = int(tcfg["batch_size"]) + self.local_rng.choice([-8, 0, 8])
        payload = {
            "learning_rate": max(5e-5, min(8e-4, lr)),
            "min_learning_rate": max(1e-5, min(2e-4, lr * 0.2)),
            "weight_decay": max(0.0, min(0.3, wd)),
            "dropout": max(0.0, min(0.35, float(tcfg["dropout"]) + self.local_rng.choice([-0.03, 0.0, 0.03]))),
            "batch_size": max(8, min(128, batch_size)),
            "micro_batch_size": max(8, min(128, batch_size)),
            "warmup_iters": max(20, min(500, int(tcfg["warmup_iters"]) + self.local_rng.choice([-25, 0, 25]))),
            "scheduler_name": self.local_rng.choice(["cosine", "linear", "none"]),
        }
        return build_patch(self, cycle_index, "hyperparam", payload, "Search bounded optimizer and schedule parameters.")

    def describe_capabilities(self) -> str:
        return "Optimizes LR, batch size, regularization, and scheduling under bounded safe ranges."


class LayerArchitectAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        payload = {
            "n_layer_delta": self.local_rng.choice([-1, 0, 1]),
            "n_head_delta": self.local_rng.choice([0, 1]),
            "n_embd_delta": self.local_rng.choice([-64, 0, 64]),
            "use_gru_gate": self.local_rng.choice([False, True]),
            "residual_scale": self.local_rng.choice([0.9, 1.0, 1.1]),
        }
        return build_patch(self, cycle_index, "model_arch", payload, "Adjust model depth/width and optional gated residual path.")

    def describe_capabilities(self) -> str:
        return "Refactors Transformer topology while preserving tensor compatibility constraints."


class TokenizerOptimizerAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        merges = [("t", "h"), ("i", "n"), ("e", "r")]
        self.local_rng.shuffle(merges)
        payload = {"merges": merges[: self.local_rng.choice([1, 2, 3])]}
        return build_patch(self, cycle_index, "tokenizer", payload, "Introduce bounded merge metadata for tokenizer experimentation.")

    def describe_capabilities(self) -> str:
        return "Explores tokenizer merge strategies without corrupting dataset alignment."


class DataAugmentorAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        payload = {
            "intensity": self.local_rng.choice([0.01, 0.03, 0.05, 0.08]),
            "pattern": self.local_rng.choice(["case_flip", "punctuation", "space_noise"]),
        }
        return build_patch(self, cycle_index, "data", payload, "Apply bounded text augmentation to improve robustness and generalization.")

    def describe_capabilities(self) -> str:
        return "Generates safe textual perturbations under strict augmentation budget."


class LossEngineerAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        use_focal = self.local_rng.choice([False, True])
        payload = {
            "loss_name": "focal" if use_focal else "cross_entropy",
            "label_smoothing": self.local_rng.choice([0.0, 0.02, 0.05, 0.08]),
            "focal_gamma": self.local_rng.choice([0.0, 1.0, 1.5, 2.0]),
        }
        return build_patch(self, cycle_index, "loss", payload, "Swap or blend loss shaping to stabilize confidence and rare-token learning.")

    def describe_capabilities(self) -> str:
        return "Designs and tunes loss functions, including focal weighting and label smoothing."


class SaboteurAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        payload = {
            "stress_noise_std": self.local_rng.choice([0.02, 0.05, 0.08, 0.12]),
            "sabotage_eval_only": True,
        }
        return build_patch(self, cycle_index, "sabotage", payload, "Inject controlled evaluation-only stress to verify resilience boundaries.")

    def describe_capabilities(self) -> str:
        return "Injects bounded faults to probe robustness without corrupting persistent checkpoints."


class ResilienceCheckerAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        current = float(runtime_state["training_config"]["stress_noise_std"])
        payload = {
            "stress_noise_std": max(0.0, min(0.15, current + self.local_rng.choice([-0.02, 0.0, 0.02]))),
        }
        return build_patch(self, cycle_index, "resilience", payload, "Tune stress-test profile to maintain robust validation envelope.")

    def describe_capabilities(self) -> str:
        return "Evaluates survivability under bounded perturbation and hardware/software noise models."

    def score_patch(self, patch: Patch, runtime_metrics: Dict[str, Any], runtime_state: Dict[str, Any]) -> PatchVote:
        score = 65.0
        reason = "Robustness-biased vote."
        if runtime_metrics.get("perplexity", 999.0) < 12.0:
            score += 10.0
        if patch.patch_type == "sabotage":
            score -= 10.0
        if patch.patch_type in {"memory", "speed", "bugfix"}:
            score += 6.0
        score += self.local_rng.uniform(-3.0, 3.0)
        approve = score >= 60.0
        return PatchVote(
            patch_id=patch.id,
            voter_id=self.agent_id,
            voter_role=self.role,
            approve=approve,
            score=max(0.0, min(100.0, score)),
            reason=reason,
        )


class SpeedDemonAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        payload = {
            "compile_model": self.local_rng.choice([False, True]),
            "num_workers": self.local_rng.choice([0, 2, 4, 6]),
            "pin_memory": True,
            "scheduler_name": self.local_rng.choice(["cosine", "linear"]),
        }
        return build_patch(self, cycle_index, "speed", payload, "Push runtime throughput with compile and loader tuning.")

    def describe_capabilities(self) -> str:
        return "Optimizes token throughput via compile, loader, and scheduling adjustments."

    def score_patch(self, patch: Patch, runtime_metrics: Dict[str, Any], runtime_state: Dict[str, Any]) -> PatchVote:
        score = 62.0 + (runtime_metrics.get("throughput_tokens_per_sec", 0.0) / 5000.0)
        if patch.patch_type == "speed":
            score += 10.0
        if patch.patch_type == "model_arch":
            score -= 6.0
        score += self.local_rng.uniform(-4.0, 4.0)
        approve = score >= 62.0
        return PatchVote(
            patch_id=patch.id,
            voter_id=self.agent_id,
            voter_role=self.role,
            approve=approve,
            score=max(0.0, min(100.0, score)),
            reason="Throughput-sensitive vote.",
        )


class MemoryWardenAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        batch_size = int(runtime_state["training_config"]["batch_size"])
        payload = {
            "use_gradient_checkpointing": self.local_rng.choice([False, True]),
            "amp_enabled": True,
            "batch_size": max(8, min(128, batch_size + self.local_rng.choice([-8, 0, 8]))),
            "micro_batch_size": max(8, min(64, batch_size)),
        }
        return build_patch(self, cycle_index, "memory", payload, "Reduce memory pressure while preserving training validity.")

    def describe_capabilities(self) -> str:
        return "Controls memory footprint using AMP, checkpointing, and bounded batch adaptation."

    def score_patch(self, patch: Patch, runtime_metrics: Dict[str, Any], runtime_state: Dict[str, Any]) -> PatchVote:
        mem = runtime_metrics.get("max_memory_allocated", 0)
        score = 60.0 + (10.0 if mem > 0 else 0.0)
        if patch.patch_type == "memory":
            score += 12.0
        score += self.local_rng.uniform(-4.0, 4.0)
        approve = score >= 60.0
        return PatchVote(
            patch_id=patch.id,
            voter_id=self.agent_id,
            voter_role=self.role,
            approve=approve,
            score=max(0.0, min(100.0, score)),
            reason="Memory-stability vote.",
        )


class EvaluatorAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        payload = {"priority_metric": self.local_rng.choice(["val_loss", "perplexity", "throughput", "bleu_like"])}
        return build_patch(self, cycle_index, "evaluation", payload, "Publish evaluation preference metadata for this cycle.")

    def describe_capabilities(self) -> str:
        return "Scores candidate patches using validation loss, perplexity, BLEU-like precision, and throughput."

    def score_patch(self, patch: Patch, runtime_metrics: Dict[str, Any], runtime_state: Dict[str, Any]) -> PatchVote:
        val_loss = float(runtime_metrics.get("val_loss", 99.0))
        perplexity = float(runtime_metrics.get("perplexity", 999.0))
        throughput = float(runtime_metrics.get("throughput_tokens_per_sec", 0.0))
        bleu_like = float(runtime_metrics.get("bleu_like", 0.0))
        base = 35.0
        if patch.patch_type in {"hyperparam", "loss", "memory", "speed"}:
            base += 12.0
        if patch.patch_type == "model_arch":
            base += 10.0
        if patch.patch_type in {"sabotage", "resilience"}:
            base -= 25.0
        val_loss_bonus = max(0.0, 45.0 - val_loss * 12.0)
        ppl_bonus = max(0.0, 8.0 - math.log(max(1.0, perplexity)) * 2.0)
        throughput_bonus = min(4.0, throughput / 20000.0)
        bleu_bonus = min(3.0, bleu_like / 50.0)
        score = base + val_loss_bonus + ppl_bonus + throughput_bonus + bleu_bonus + self.local_rng.uniform(-1.0, 1.0)
        approve = score >= 72.0 and patch.patch_type not in {"sabotage", "resilience"}
        return PatchVote(
            patch_id=patch.id,
            voter_id=self.agent_id,
            voter_role=self.role,
            approve=approve,
            score=max(0.0, min(100.0, score)),
            reason="Val-loss-dominant evaluator vote.",
        )


class ArbiterAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        payload = {"max_patches": runtime_state["swarm_config"]["max_patches_applied_per_cycle"]}
        return build_patch(self, cycle_index, "arbitration", payload, "Declare arbitration capacity and patch conflict resolution policy.")

    def describe_capabilities(self) -> str:
        return "Resolves conflicts, computes weighted voting, applies patches, and manages lifecycle events."


def build_patch(agent: Agent, cycle_index: int, patch_type: str, payload: Dict[str, Any], summary: str) -> Patch:
    patch_id = stable_hash({
        "agent_id": agent.agent_id,
        "role": agent.role,
        "cycle_index": cycle_index,
        "patch_type": patch_type,
        "payload": payload,
    })[:24]
    return Patch(
        id=patch_id,
        author_id=agent.agent_id,
        role=agent.role,
        timestamp=utc_now(),
        patch_type=patch_type,
        payload=payload,
        summary=summary,
        cycle_index=cycle_index,
    )


ROLE_TO_CLASS: Dict[str, type[Agent]] = {
    "BugHunter": BugHunterAgent,
    "HyperparamTuner": HyperparamTunerAgent,
    "LayerArchitect": LayerArchitectAgent,
    "TokenizerOptimizer": TokenizerOptimizerAgent,
    "DataAugmentor": DataAugmentorAgent,
    "LossEngineer": LossEngineerAgent,
    "Saboteur": SaboteurAgent,
    "ResilienceChecker": ResilienceCheckerAgent,
    "SpeedDemon": SpeedDemonAgent,
    "MemoryWarden": MemoryWardenAgent,
    "Evaluator": EvaluatorAgent,
    "Arbiter": ArbiterAgent,
}


class LocalAgentWorker:
    def __init__(self, agent: Agent):
        self.agent = agent

    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        return self.agent.generate_patch(runtime_state, cycle_index)

    def score_patch(self, patch: Patch, runtime_metrics: Dict[str, Any], runtime_state: Dict[str, Any]) -> PatchVote:
        return self.agent.score_patch(patch, runtime_metrics, runtime_state)

    def get_state(self) -> AgentState:
        return self.agent.state


__all__ = [
    "AgentState",
    "Agent",
    "BugHunterAgent",
    "HyperparamTunerAgent",
    "LayerArchitectAgent",
    "TokenizerOptimizerAgent",
    "DataAugmentorAgent",
    "LossEngineerAgent",
    "SaboteurAgent",
    "ResilienceCheckerAgent",
    "SpeedDemonAgent",
    "MemoryWardenAgent",
    "EvaluatorAgent",
    "ArbiterAgent",
    "build_patch",
    "ROLE_TO_CLASS",
    "LocalAgentWorker",
]
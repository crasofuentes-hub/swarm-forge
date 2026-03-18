"""Patch governance primitives for Swarm Forge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .core import PATCH_TYPES, stable_hash


@dataclass
class Patch:
    id: str
    author_id: str
    role: str
    timestamp: str
    patch_type: str
    payload: Dict[str, Any]
    summary: str
    cycle_index: int

    def payload_hash(self) -> str:
        return stable_hash(self.payload)


@dataclass
class PatchVote:
    patch_id: str
    voter_id: str
    voter_role: str
    approve: bool
    score: float
    reason: str
    weight: float = 0.0


@dataclass
class PatchDecision:
    patch_id: str
    approval_percent: float
    weighted_score: float
    applied: bool
    reason: str
    conflict_group: str
    summary: str
    author_id: str
    role: str
    patch_type: str


class PatchConflictResolver:
    @staticmethod
    def conflict_group(patch: Patch) -> str:
        if patch.patch_type in {"model_arch", "tokenizer"}:
            return patch.patch_type
        if patch.patch_type in {"hyperparam", "loss", "memory", "speed"}:
            return "training_config"
        if patch.patch_type in {"sabotage", "resilience"}:
            return "robustness"
        if patch.patch_type == "data":
            return "data"
        return f"misc:{patch.patch_type}"


__all__ = [
    "Patch",
    "PatchVote",
    "PatchDecision",
    "PatchConflictResolver",
]
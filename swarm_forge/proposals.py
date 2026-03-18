"""Experiment proposal models for Swarm Forge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .core import stable_hash, utc_now


@dataclass
class AppliedPatchRecord:
    patch_id: str
    patch_type: str
    author_id: str
    role: str
    applied_at: str
    payload_hash: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentProposal:
    proposal_id: str
    author_id: str
    author_role: str
    timestamp: str
    dataset_name: str
    hypothesis: str
    changed_variable: str
    proposed_value: Any
    success_metric: str
    success_threshold: float
    rollback_condition: str
    notes: str = ""


def build_experiment_proposal(
    author_id: str,
    author_role: str,
    dataset_name: str,
    hypothesis: str,
    changed_variable: str,
    proposed_value: Any,
    success_metric: str,
    success_threshold: float,
    rollback_condition: str,
    notes: str = "",
) -> ExperimentProposal:
    timestamp = utc_now()
    payload = {
        "author_id": author_id,
        "author_role": author_role,
        "dataset_name": dataset_name,
        "hypothesis": hypothesis,
        "changed_variable": changed_variable,
        "proposed_value": proposed_value,
        "success_metric": success_metric,
        "success_threshold": success_threshold,
        "rollback_condition": rollback_condition,
        "notes": notes,
    }
    proposal_id = stable_hash(payload)[:24]
    return ExperimentProposal(
        proposal_id=proposal_id,
        author_id=author_id,
        author_role=author_role,
        timestamp=timestamp,
        dataset_name=dataset_name,
        hypothesis=hypothesis,
        changed_variable=changed_variable,
        proposed_value=proposed_value,
        success_metric=success_metric,
        success_threshold=float(success_threshold),
        rollback_condition=rollback_condition,
        notes=notes,
    )


__all__ = [
    "AppliedPatchRecord",
    "ExperimentProposal",
    "build_experiment_proposal",
]
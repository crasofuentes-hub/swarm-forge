"""Research orchestration contracts and scoring helpers for Swarm Forge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .proposals import ExperimentProposal


@dataclass
class CampaignConfig:
    campaign_id: str
    dataset_name: str
    objective_metric: str = "val_loss"
    maximize: bool = False
    notes: str = ""


@dataclass
class TrialSpec:
    trial_id: str
    campaign_id: str
    hypothesis: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class TrialResult:
    trial_id: str
    campaign_id: str
    success: bool
    objective_metric: str
    objective_value: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    notes: str = ""


@dataclass
class CampaignSummary:
    campaign_id: str
    objective_metric: str
    best_trial_id: Optional[str]
    best_objective_value: Optional[float]
    total_trials: int
    successful_trials: int


def score_trial_result(result: TrialResult, maximize: bool = False) -> float:
    value = float(result.objective_value)
    return value if maximize else -value


def select_best_trial(results: List[TrialResult], maximize: bool = False) -> Optional[TrialResult]:
    successful = [r for r in results if r.success]
    if not successful:
        return None
    return max(successful, key=lambda r: score_trial_result(r, maximize=maximize))


def build_campaign_summary(
    campaign_id: str,
    objective_metric: str,
    results: List[TrialResult],
    maximize: bool = False,
) -> CampaignSummary:
    best = select_best_trial(results, maximize=maximize)
    successful_trials = sum(1 for r in results if r.success)
    return CampaignSummary(
        campaign_id=campaign_id,
        objective_metric=objective_metric,
        best_trial_id=best.trial_id if best is not None else None,
        best_objective_value=best.objective_value if best is not None else None,
        total_trials=len(results),
        successful_trials=successful_trials,
    )
def proposal_to_trial_spec(proposal: ExperimentProposal, campaign_id: Optional[str] = None) -> TrialSpec:
    resolved_campaign_id = campaign_id or f"{proposal.dataset_name}:{proposal.success_metric}"
    overrides = {
        proposal.changed_variable: proposal.proposed_value,
    }
    tags = [
        proposal.dataset_name,
        proposal.author_role,
        proposal.changed_variable,
    ]
    return TrialSpec(
        trial_id=proposal.proposal_id,
        campaign_id=resolved_campaign_id,
        hypothesis=proposal.hypothesis,
        overrides=overrides,
        tags=tags,
    )


def proposals_to_trial_specs(
    proposals: List[ExperimentProposal],
    campaign_id: Optional[str] = None,
) -> List[TrialSpec]:
    return [proposal_to_trial_spec(p, campaign_id=campaign_id) for p in proposals]
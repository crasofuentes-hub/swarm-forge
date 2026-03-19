"""Search-state contracts for experimental search in Swarm Forge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .research import TrialSpec


@dataclass
class SearchState:
    campaign_id: str
    state_id: str
    dataset_name: str
    objective_metric: str
    maximize: bool
    base_trial_id: Optional[str] = None
    applied_overrides: Dict[str, Any] = field(default_factory=dict)
    budget_remaining: int = 0
    depth: int = 0
    parent_state_id: Optional[str] = None
    last_trial_id: Optional[str] = None
    last_objective_value: Optional[float] = None


@dataclass
class SearchAction:
    action_id: str
    action_type: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    source_proposal_id: Optional[str] = None
    description: str = ""


def apply_action_to_state(
    state: SearchState,
    action: SearchAction,
    next_state_id: str,
) -> SearchState:
    merged_overrides = dict(state.applied_overrides)
    merged_overrides.update(action.overrides)

    return SearchState(
        campaign_id=state.campaign_id,
        state_id=next_state_id,
        dataset_name=state.dataset_name,
        objective_metric=state.objective_metric,
        maximize=state.maximize,
        base_trial_id=state.base_trial_id,
        applied_overrides=merged_overrides,
        budget_remaining=max(0, state.budget_remaining - 1),
        depth=state.depth + 1,
        parent_state_id=state.state_id,
        last_trial_id=state.last_trial_id,
        last_objective_value=state.last_objective_value,
    )
def search_state_to_trial_spec(
    state: SearchState,
    trial_id: str,
    hypothesis: str,
) -> TrialSpec:
    return TrialSpec(
        trial_id=trial_id,
        campaign_id=state.campaign_id,
        hypothesis=hypothesis,
        overrides=dict(state.applied_overrides),
        tags=[
            state.dataset_name,
            state.objective_metric,
            f"depth:{state.depth}",
            f"state:{state.state_id}",
        ],
    )


def search_transition_to_trial_spec(
    state: SearchState,
    action: SearchAction,
    next_state_id: str,
    hypothesis: str,
) -> TrialSpec:
    next_state = apply_action_to_state(state, action, next_state_id=next_state_id)
    return search_state_to_trial_spec(
        state=next_state,
        trial_id=next_state.state_id,
        hypothesis=hypothesis,
    )
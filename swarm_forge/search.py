"""Search-state contracts for experimental search in Swarm Forge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .research import TrialSpec
from .proposals import ExperimentProposal


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
class SearchSession:
    def __init__(self, root_state: SearchState):
        self.root_state = root_state
        self.states: Dict[str, SearchState] = {root_state.state_id: root_state}
        self.actions_by_state: Dict[str, list[SearchAction]] = {}

    def get_state(self, state_id: str) -> SearchState:
        return self.states[state_id]

    def apply_action(self, state_id: str, action: SearchAction, next_state_id: str) -> SearchState:
        state = self.get_state(state_id)
        next_state = apply_action_to_state(state, action, next_state_id=next_state_id)
        self.states[next_state.state_id] = next_state
        self.actions_by_state.setdefault(state_id, []).append(action)
        return next_state

    def state_to_trial_spec(self, state_id: str, trial_id: str, hypothesis: str) -> TrialSpec:
        state = self.get_state(state_id)
        return search_state_to_trial_spec(
            state=state,
            trial_id=trial_id,
            hypothesis=hypothesis,
        )

    def transition_to_trial_spec(
        self,
        state_id: str,
        action: SearchAction,
        next_state_id: str,
        hypothesis: str,
    ) -> TrialSpec:
        next_state = self.apply_action(state_id, action, next_state_id=next_state_id)
        return search_state_to_trial_spec(
            state=next_state,
            trial_id=next_state.state_id,
            hypothesis=hypothesis,
        )
def proposal_to_search_action(proposal: ExperimentProposal) -> SearchAction:
    return SearchAction(
        action_id=proposal.proposal_id,
        action_type="proposal_override",
        overrides={
            proposal.changed_variable: proposal.proposed_value,
        },
        source_proposal_id=proposal.proposal_id,
        description=proposal.hypothesis,
    )


def proposals_to_search_actions(proposals: list[ExperimentProposal]) -> list[SearchAction]:
    return [proposal_to_search_action(p) for p in proposals]
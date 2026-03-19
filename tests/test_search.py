import swarm_forge as sf


def test_apply_action_to_state_merges_overrides_and_advances_depth():
    state = sf.SearchState(
        campaign_id="camp-search-1",
        state_id="s0",
        dataset_name="tinyshakespeare",
        objective_metric="val_loss",
        maximize=False,
        applied_overrides={"learning_rate": 3e-4},
        budget_remaining=5,
        depth=0,
    )

    action = sf.SearchAction(
        action_id="a1",
        action_type="train_override",
        overrides={"learning_rate": 5e-5, "batch_size": 8},
        source_proposal_id="prop-001",
        description="shrink lr and batch",
    )

    next_state = sf.apply_action_to_state(state, action, next_state_id="s1")

    assert next_state.state_id == "s1"
    assert next_state.parent_state_id == "s0"
    assert next_state.depth == 1
    assert next_state.budget_remaining == 4
    assert next_state.applied_overrides["learning_rate"] == 5e-5
    assert next_state.applied_overrides["batch_size"] == 8


def test_apply_action_to_state_never_makes_negative_budget():
    state = sf.SearchState(
        campaign_id="camp-search-2",
        state_id="s0",
        dataset_name="tinyshakespeare",
        objective_metric="val_loss",
        maximize=False,
        budget_remaining=0,
    )

    action = sf.SearchAction(
        action_id="a1",
        action_type="train_override",
        overrides={"learning_rate": 1e-4},
    )

    next_state = sf.apply_action_to_state(state, action, next_state_id="s1")

    assert next_state.budget_remaining == 0
    assert next_state.depth == 1


def test_search_state_and_action_exports_exist():
    assert hasattr(sf, "SearchState")
    assert hasattr(sf, "SearchAction")
    assert hasattr(sf, "apply_action_to_state")
def test_search_state_to_trial_spec_mapping():
    state = sf.SearchState(
        campaign_id="camp-search-map",
        state_id="s1",
        dataset_name="tinyshakespeare",
        objective_metric="val_loss",
        maximize=False,
        applied_overrides={"learning_rate": 5e-5, "batch_size": 8},
        budget_remaining=3,
        depth=1,
    )

    trial = sf.search_state_to_trial_spec(
        state=state,
        trial_id="trial-from-state-001",
        hypothesis="evaluate mapped state",
    )

    assert trial.trial_id == "trial-from-state-001"
    assert trial.campaign_id == "camp-search-map"
    assert trial.overrides["learning_rate"] == 5e-5
    assert trial.overrides["batch_size"] == 8
    assert "tinyshakespeare" in trial.tags
    assert "val_loss" in trial.tags


def test_search_transition_to_trial_spec_applies_action_first():
    state = sf.SearchState(
        campaign_id="camp-search-transition",
        state_id="s0",
        dataset_name="tinyshakespeare",
        objective_metric="val_loss",
        maximize=False,
        applied_overrides={"learning_rate": 3e-4},
        budget_remaining=4,
        depth=0,
    )

    action = sf.SearchAction(
        action_id="a1",
        action_type="train_override",
        overrides={"learning_rate": 1e-4, "batch_size": 16},
        description="refine train regime",
    )

    trial = sf.search_transition_to_trial_spec(
        state=state,
        action=action,
        next_state_id="s1",
        hypothesis="apply transition then execute trial",
    )

    assert trial.trial_id == "s1"
    assert trial.campaign_id == "camp-search-transition"
    assert trial.overrides["learning_rate"] == 1e-4
    assert trial.overrides["batch_size"] == 16
    assert "depth:1" in trial.tags
    assert "state:s1" in trial.tags
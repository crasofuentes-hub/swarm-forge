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
def test_search_session_apply_action_registers_new_state():
    root = sf.SearchState(
        campaign_id="camp-session-1",
        state_id="root",
        dataset_name="tinyshakespeare",
        objective_metric="val_loss",
        maximize=False,
        applied_overrides={"learning_rate": 3e-4},
        budget_remaining=5,
        depth=0,
    )

    session = sf.SearchSession(root)

    action = sf.SearchAction(
        action_id="a1",
        action_type="train_override",
        overrides={"learning_rate": 1e-4, "batch_size": 8},
        description="refine training setup",
    )

    next_state = session.apply_action("root", action, next_state_id="s1")

    assert next_state.state_id == "s1"
    assert next_state.parent_state_id == "root"
    assert next_state.depth == 1
    assert session.get_state("s1").applied_overrides["learning_rate"] == 1e-4
    assert len(session.actions_by_state["root"]) == 1


def test_search_session_state_to_trial_spec():
    root = sf.SearchState(
        campaign_id="camp-session-2",
        state_id="root",
        dataset_name="tinyshakespeare",
        objective_metric="val_loss",
        maximize=False,
        applied_overrides={"learning_rate": 5e-5},
        budget_remaining=3,
        depth=0,
    )

    session = sf.SearchSession(root)
    trial = session.state_to_trial_spec("root", trial_id="trial-root", hypothesis="run root state")

    assert trial.trial_id == "trial-root"
    assert trial.campaign_id == "camp-session-2"
    assert trial.overrides["learning_rate"] == 5e-5


def test_search_session_transition_to_trial_spec():
    root = sf.SearchState(
        campaign_id="camp-session-3",
        state_id="root",
        dataset_name="tinyshakespeare",
        objective_metric="val_loss",
        maximize=False,
        applied_overrides={"learning_rate": 3e-4},
        budget_remaining=4,
        depth=0,
    )

    session = sf.SearchSession(root)
    action = sf.SearchAction(
        action_id="a1",
        action_type="train_override",
        overrides={"learning_rate": 1e-4},
        description="lower lr",
    )

    trial = session.transition_to_trial_spec(
        state_id="root",
        action=action,
        next_state_id="s1",
        hypothesis="execute next search state",
    )

    assert trial.trial_id == "s1"
    assert trial.overrides["learning_rate"] == 1e-4
    assert session.get_state("s1").depth == 1
def test_proposal_to_search_action_mapping():
    proposal = sf.ExperimentProposal(
        proposal_id="prop-search-001",
        author_id="HyperparamTuner-00",
        author_role="HyperparamTuner",
        timestamp="2026-03-19T00:00:00Z",
        dataset_name="tinyshakespeare",
        hypothesis="lower learning rate for stability",
        changed_variable="learning_rate",
        proposed_value=5e-5,
        success_metric="val_loss",
        success_threshold=2.5,
        rollback_condition="val_loss > 2.7",
    )

    action = sf.proposal_to_search_action(proposal)

    assert action.action_id == "prop-search-001"
    assert action.action_type == "proposal_override"
    assert action.overrides["learning_rate"] == 5e-5
    assert action.source_proposal_id == "prop-search-001"
    assert "learning rate" in action.description


def test_proposals_to_search_actions_batch():
    proposals = [
        sf.ExperimentProposal(
            proposal_id="prop-search-001",
            author_id="HyperparamTuner-00",
            author_role="HyperparamTuner",
            timestamp="2026-03-19T00:00:00Z",
            dataset_name="tinyshakespeare",
            hypothesis="lower learning rate",
            changed_variable="learning_rate",
            proposed_value=5e-5,
            success_metric="val_loss",
            success_threshold=2.5,
            rollback_condition="val_loss > 2.7",
        ),
        sf.ExperimentProposal(
            proposal_id="prop-search-002",
            author_id="LossEngineer-00",
            author_role="LossEngineer",
            timestamp="2026-03-19T00:01:00Z",
            dataset_name="tinyshakespeare",
            hypothesis="small label smoothing",
            changed_variable="label_smoothing",
            proposed_value=0.02,
            success_metric="val_loss",
            success_threshold=2.5,
            rollback_condition="val_loss > 2.7",
        ),
    ]

    actions = sf.proposals_to_search_actions(proposals)

    assert len(actions) == 2
    assert actions[0].source_proposal_id == "prop-search-001"
    assert actions[1].overrides["label_smoothing"] == 0.02
def test_search_session_expand_proposals_to_trials():
    root = sf.SearchState(
        campaign_id="camp-expand-1",
        state_id="root",
        dataset_name="tinyshakespeare",
        objective_metric="val_loss",
        maximize=False,
        applied_overrides={"learning_rate": 3e-4},
        budget_remaining=5,
        depth=0,
    )

    session = sf.SearchSession(root)

    proposals = [
        sf.ExperimentProposal(
            proposal_id="prop-a",
            author_id="HyperparamTuner-00",
            author_role="HyperparamTuner",
            timestamp="2026-03-19T00:00:00Z",
            dataset_name="tinyshakespeare",
            hypothesis="lower lr",
            changed_variable="learning_rate",
            proposed_value=5e-5,
            success_metric="val_loss",
            success_threshold=2.5,
            rollback_condition="val_loss > 2.7",
        ),
        sf.ExperimentProposal(
            proposal_id="prop-b",
            author_id="LossEngineer-00",
            author_role="LossEngineer",
            timestamp="2026-03-19T00:01:00Z",
            dataset_name="tinyshakespeare",
            hypothesis="small smoothing",
            changed_variable="label_smoothing",
            proposed_value=0.02,
            success_metric="val_loss",
            success_threshold=2.5,
            rollback_condition="val_loss > 2.7",
        ),
    ]

    trials = session.expand_proposals_to_trials("root", proposals, hypothesis_prefix="proposal batch")

    assert len(trials) == 2
    assert trials[0].campaign_id == "camp-expand-1"
    assert trials[0].trial_id == "root__prop-a"
    assert trials[1].trial_id == "root__prop-b"
    assert trials[0].overrides["learning_rate"] == 5e-5
    assert trials[1].overrides["label_smoothing"] == 0.02
    assert session.get_state("root__prop-a").depth == 1
    assert session.get_state("root__prop-b").depth == 1
    assert len(session.actions_by_state["root"]) == 2


def test_search_session_expand_proposals_to_trials_keeps_prior_overrides():
    root = sf.SearchState(
        campaign_id="camp-expand-2",
        state_id="root",
        dataset_name="tinyshakespeare",
        objective_metric="val_loss",
        maximize=False,
        applied_overrides={"batch_size": 8},
        budget_remaining=3,
        depth=0,
    )

    session = sf.SearchSession(root)

    proposals = [
        sf.ExperimentProposal(
            proposal_id="prop-c",
            author_id="HyperparamTuner-00",
            author_role="HyperparamTuner",
            timestamp="2026-03-19T00:02:00Z",
            dataset_name="tinyshakespeare",
            hypothesis="set lr",
            changed_variable="learning_rate",
            proposed_value=1e-4,
            success_metric="val_loss",
            success_threshold=2.5,
            rollback_condition="val_loss > 2.7",
        )
    ]

    trials = session.expand_proposals_to_trials("root", proposals)

    assert len(trials) == 1
    assert trials[0].overrides["batch_size"] == 8
    assert trials[0].overrides["learning_rate"] == 1e-4
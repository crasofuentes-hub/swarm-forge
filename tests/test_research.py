import swarm_forge as sf


def test_select_best_trial_for_min_objective():
    results = [
        sf.TrialResult(
            trial_id="t1",
            campaign_id="c1",
            success=True,
            objective_metric="val_loss",
            objective_value=2.80,
        ),
        sf.TrialResult(
            trial_id="t2",
            campaign_id="c1",
            success=True,
            objective_metric="val_loss",
            objective_value=2.56,
        ),
        sf.TrialResult(
            trial_id="t3",
            campaign_id="c1",
            success=False,
            objective_metric="val_loss",
            objective_value=2.10,
        ),
    ]

    best = sf.select_best_trial(results, maximize=False)

    assert best is not None
    assert best.trial_id == "t2"
    assert best.objective_value == 2.56


def test_select_best_trial_for_max_objective():
    results = [
        sf.TrialResult(
            trial_id="t1",
            campaign_id="c1",
            success=True,
            objective_metric="bleu_like",
            objective_value=12.0,
        ),
        sf.TrialResult(
            trial_id="t2",
            campaign_id="c1",
            success=True,
            objective_metric="bleu_like",
            objective_value=18.5,
        ),
    ]

    best = sf.select_best_trial(results, maximize=True)

    assert best is not None
    assert best.trial_id == "t2"


def test_build_campaign_summary():
    results = [
        sf.TrialResult(
            trial_id="t1",
            campaign_id="camp-alpha",
            success=True,
            objective_metric="val_loss",
            objective_value=2.70,
        ),
        sf.TrialResult(
            trial_id="t2",
            campaign_id="camp-alpha",
            success=True,
            objective_metric="val_loss",
            objective_value=2.54,
        ),
        sf.TrialResult(
            trial_id="t3",
            campaign_id="camp-alpha",
            success=False,
            objective_metric="val_loss",
            objective_value=2.40,
        ),
    ]

    summary = sf.build_campaign_summary(
        campaign_id="camp-alpha",
        objective_metric="val_loss",
        results=results,
        maximize=False,
    )

    assert summary.campaign_id == "camp-alpha"
    assert summary.best_trial_id == "t2"
    assert summary.best_objective_value == 2.54
    assert summary.total_trials == 3
    assert summary.successful_trials == 2
def test_proposal_to_trial_spec_mapping():
    proposal = sf.ExperimentProposal(
        proposal_id="prop-001",
        author_id="HyperparamTuner-00",
        author_role="HyperparamTuner",
        timestamp="2026-03-18T00:00:00Z",
        dataset_name="wikitext2",
        hypothesis="Smaller LR may stabilize resumed optimization.",
        changed_variable="learning_rate",
        proposed_value=5e-5,
        success_metric="val_loss",
        success_threshold=2.56,
        rollback_condition="val_loss > 2.60",
        notes="bounded resume-reset sweep",
    )

    trial = sf.proposal_to_trial_spec(proposal)

    assert trial.trial_id == "prop-001"
    assert trial.campaign_id == "wikitext2:val_loss"
    assert trial.overrides["learning_rate"] == 5e-5
    assert "wikitext2" in trial.tags
    assert "HyperparamTuner" in trial.tags


def test_proposals_to_trial_specs_batch():
    proposals = [
        sf.ExperimentProposal(
            proposal_id="prop-001",
            author_id="HyperparamTuner-00",
            author_role="HyperparamTuner",
            timestamp="2026-03-18T00:00:00Z",
            dataset_name="wikitext2",
            hypothesis="LR adjustment",
            changed_variable="learning_rate",
            proposed_value=5e-5,
            success_metric="val_loss",
            success_threshold=2.56,
            rollback_condition="val_loss > 2.60",
        ),
        sf.ExperimentProposal(
            proposal_id="prop-002",
            author_id="LossEngineer-00",
            author_role="LossEngineer",
            timestamp="2026-03-18T00:01:00Z",
            dataset_name="wikitext2",
            hypothesis="Try light label smoothing",
            changed_variable="label_smoothing",
            proposed_value=0.02,
            success_metric="val_loss",
            success_threshold=2.56,
            rollback_condition="val_loss > 2.60",
        ),
    ]

    trials = sf.proposals_to_trial_specs(proposals, campaign_id="camp-wiki-a")

    assert len(trials) == 2
    assert trials[0].campaign_id == "camp-wiki-a"
    assert trials[1].campaign_id == "camp-wiki-a"
    assert trials[1].overrides["label_smoothing"] == 0.02
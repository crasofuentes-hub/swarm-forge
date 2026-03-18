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
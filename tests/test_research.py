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
def test_campaign_runner_summary_and_best_result():
    runner = sf.CampaignRunner(
        sf.CampaignConfig(
            campaign_id="camp-wiki-a",
            dataset_name="wikitext2",
            objective_metric="val_loss",
            maximize=False,
        )
    )

    runner.add_trial(
        sf.TrialSpec(
            trial_id="t1",
            campaign_id="camp-wiki-a",
            hypothesis="baseline",
            overrides={"learning_rate": 3e-4},
        )
    )
    runner.add_trial(
        sf.TrialSpec(
            trial_id="t2",
            campaign_id="camp-wiki-a",
            hypothesis="smaller lr",
            overrides={"learning_rate": 5e-5},
        )
    )

    runner.add_result(
        sf.TrialResult(
            trial_id="t1",
            campaign_id="camp-wiki-a",
            success=True,
            objective_metric="val_loss",
            objective_value=2.70,
        )
    )
    runner.add_result(
        sf.TrialResult(
            trial_id="t2",
            campaign_id="camp-wiki-a",
            success=True,
            objective_metric="val_loss",
            objective_value=2.56,
        )
    )

    best = runner.best_result()
    summary = runner.summary()

    assert best is not None
    assert best.trial_id == "t2"
    assert summary.best_trial_id == "t2"
    assert summary.best_objective_value == 2.56
    assert summary.total_trials == 2
    assert summary.successful_trials == 2


def test_campaign_runner_rejects_mismatched_trial_campaign_id():
    runner = sf.CampaignRunner(
        sf.CampaignConfig(
            campaign_id="camp-main",
            dataset_name="wikitext2",
        )
    )

    try:
        runner.add_trial(
            sf.TrialSpec(
                trial_id="t-x",
                campaign_id="camp-other",
                hypothesis="bad",
                overrides={},
            )
        )
        assert False, "Expected ValueError for mismatched campaign_id"
    except ValueError as exc:
        assert "campaign_id" in str(exc)


def test_campaign_runner_rejects_mismatched_result_campaign_id():
    runner = sf.CampaignRunner(
        sf.CampaignConfig(
            campaign_id="camp-main",
            dataset_name="wikitext2",
        )
    )

    try:
        runner.add_result(
            sf.TrialResult(
                trial_id="r-x",
                campaign_id="camp-other",
                success=True,
                objective_metric="val_loss",
                objective_value=2.5,
            )
        )
        assert False, "Expected ValueError for mismatched campaign_id"
    except ValueError as exc:
        assert "campaign_id" in str(exc)
def test_trial_executor_runs_local_trial_and_returns_result(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "input.txt").write_text(("To be, or not to be.\n" * 300), encoding="utf-8")

    campaign = sf.CampaignConfig(
        campaign_id="camp-exec-1",
        dataset_name="tinyshakespeare",
        objective_metric="val_loss",
        maximize=False,
    )

    base_train_cfg = sf.TrainingConfig(
        device="cpu",
        batch_size=4,
        micro_batch_size=4,
        block_size=32,
        max_iters_per_cycle=2,
        eval_iters=2,
        amp_enabled=False,
        patch_trial_train_steps=1,
    )
    base_model_cfg = sf.ModelConfig(
        vocab_size=256,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.1,
    )

    executor = sf.TrialExecutor(
        campaign=campaign,
        base_train_cfg=base_train_cfg,
        base_model_cfg=base_model_cfg,
        data_dir=str(data_dir),
        output_root=str(tmp_path / "runs"),
    )

    trial = sf.TrialSpec(
        trial_id="trial-001",
        campaign_id="camp-exec-1",
        hypothesis="lower lr quick probe",
        overrides={"learning_rate": 5e-5, "patch_trial_train_steps": 1},
    )

    result = executor.execute(trial)

    assert result.trial_id == "trial-001"
    assert result.campaign_id == "camp-exec-1"
    assert result.success is True
    assert result.objective_metric == "val_loss"
    assert isinstance(result.objective_value, float)
    assert result.metrics["initial"]["val_loss"] >= 0.0
    assert result.metrics["final"]["val_loss"] >= 0.0
    assert result.metrics["train"]["global_step"] >= 1
    assert result.checkpoint_path is not None


def test_trial_executor_applies_training_overrides_only():
    campaign = sf.CampaignConfig(
        campaign_id="camp-exec-2",
        dataset_name="tinyshakespeare",
    )

    base_train_cfg = sf.TrainingConfig(device="cpu", amp_enabled=False)
    base_model_cfg = sf.ModelConfig()

    executor = sf.TrialExecutor(
        campaign=campaign,
        base_train_cfg=base_train_cfg,
        base_model_cfg=base_model_cfg,
        data_dir="data/tinyshakespeare",
        output_root="runs/test-exec",
    )

    trial = sf.TrialSpec(
        trial_id="trial-002",
        campaign_id="camp-exec-2",
        hypothesis="override train-only fields",
        overrides={"learning_rate": 1e-4, "batch_size": 8, "n_layer": 99},
    )

    train_cfg = executor._build_trial_train_cfg(trial)
    model_cfg = executor._build_trial_model_cfg()

    assert train_cfg.learning_rate == 1e-4
    assert train_cfg.batch_size == 8
    assert model_cfg.n_layer == base_model_cfg.n_layer
def test_campaign_runner_run_trial_executes_and_records_result(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "input.txt").write_text(("To be, or not to be.\n" * 300), encoding="utf-8")

    campaign = sf.CampaignConfig(
        campaign_id="camp-runner-exec-1",
        dataset_name="tinyshakespeare",
        objective_metric="val_loss",
        maximize=False,
    )

    runner = sf.CampaignRunner(campaign)

    executor = sf.TrialExecutor(
        campaign=campaign,
        base_train_cfg=sf.TrainingConfig(
            device="cpu",
            batch_size=4,
            micro_batch_size=4,
            block_size=32,
            max_iters_per_cycle=2,
            eval_iters=2,
            amp_enabled=False,
            patch_trial_train_steps=1,
        ),
        base_model_cfg=sf.ModelConfig(
            vocab_size=256,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.1,
        ),
        data_dir=str(data_dir),
        output_root=str(tmp_path / "runs"),
    )

    trial = sf.TrialSpec(
        trial_id="trial-runner-001",
        campaign_id="camp-runner-exec-1",
        hypothesis="runner executes one trial",
        overrides={"learning_rate": 5e-5, "patch_trial_train_steps": 1},
    )

    result = runner.run_trial(executor, trial)
    summary = runner.summary()

    assert result.trial_id == "trial-runner-001"
    assert len(runner.trials) == 1
    assert len(runner.results) == 1
    assert summary.total_trials == 1
    assert summary.successful_trials == 1
    assert summary.best_trial_id == "trial-runner-001"


def test_campaign_runner_run_trials_executes_batch(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "input.txt").write_text(("To be, or not to be.\n" * 300), encoding="utf-8")

    campaign = sf.CampaignConfig(
        campaign_id="camp-runner-exec-2",
        dataset_name="tinyshakespeare",
        objective_metric="val_loss",
        maximize=False,
    )

    runner = sf.CampaignRunner(campaign)

    executor = sf.TrialExecutor(
        campaign=campaign,
        base_train_cfg=sf.TrainingConfig(
            device="cpu",
            batch_size=4,
            micro_batch_size=4,
            block_size=32,
            max_iters_per_cycle=2,
            eval_iters=2,
            amp_enabled=False,
            patch_trial_train_steps=1,
        ),
        base_model_cfg=sf.ModelConfig(
            vocab_size=256,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.1,
        ),
        data_dir=str(data_dir),
        output_root=str(tmp_path / "runs"),
    )

    trials = [
        sf.TrialSpec(
            trial_id="trial-batch-001",
            campaign_id="camp-runner-exec-2",
            hypothesis="trial one",
            overrides={"learning_rate": 5e-5, "patch_trial_train_steps": 1},
        ),
        sf.TrialSpec(
            trial_id="trial-batch-002",
            campaign_id="camp-runner-exec-2",
            hypothesis="trial two",
            overrides={"learning_rate": 1e-4, "patch_trial_train_steps": 1},
        ),
    ]

    results = runner.run_trials(executor, trials)
    summary = runner.summary()

    assert len(results) == 2
    assert len(runner.trials) == 2
    assert len(runner.results) == 2
    assert summary.total_trials == 2
    assert summary.successful_trials == 2
    assert summary.best_trial_id in {"trial-batch-001", "trial-batch-002"}
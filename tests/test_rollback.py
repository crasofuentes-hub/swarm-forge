from pathlib import Path

import swarm_forge as sf


def test_patch_application_rollback_restores_state(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    text = ("To be, or not to be, that is the question.\n" * 400)
    (data_dir / "input.txt").write_text(text, encoding="utf-8")

    train_cfg = sf.TrainingConfig(
        device="cpu",
        batch_size=4,
        micro_batch_size=4,
        block_size=32,
        max_iters_per_cycle=1,
        eval_iters=1,
        amp_enabled=False,
    )
    model_cfg = sf.ModelConfig(
        vocab_size=256,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.1,
    )
    logger = sf.logging.getLogger("swarm_forge_test_rollback")
    dataset = sf.TinyShakespeareData(str(data_dir), train_cfg, logger)
    model_cfg.vocab_size = dataset.tokenizer.vocab_size

    runtime = sf.TrainingRuntime(
        tcfg=train_cfg,
        mcfg=model_cfg,
        dataset=dataset,
        output_dir=tmp_path / "runs",
        logger=logger,
    )
    applier = sf.PatchApplier(
        runtime=runtime,
        logger=logger,
        swarm_cfg=sf.SwarmConfig(
            output_dir=str(tmp_path / "runs2"),
            data_dir=str(data_dir),
            max_hours=1.0,
        ),
    )

    original_loss_name = runtime.loss_name
    original_batch_size = runtime.tcfg.batch_size
    original_model_config = sf.asdict(runtime.mcfg)

    patch = sf.Patch(
        id="rollback-test-patch",
        author_id="TokenizerOptimizer-00",
        role="TokenizerOptimizer",
        timestamp=sf.utc_now(),
        patch_type="tokenizer",
        payload={"merges": "this-should-be-a-list"},
        summary="force tokenizer failure to verify rollback",
        cycle_index=1,
    )

    result = applier.apply_patch(patch)

    assert result.success is False
    assert runtime.loss_name == original_loss_name
    assert runtime.tcfg.batch_size == original_batch_size
    assert sf.asdict(runtime.mcfg) == original_model_config
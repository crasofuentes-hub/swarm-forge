from pathlib import Path

import swarm_forge as sf


def test_smoke_data_and_model_init(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    text = ("To be, or not to be, that is the question.\n" * 400)
    (data_dir / "input.txt").write_text(text, encoding="utf-8")

    train_cfg = sf.TrainingConfig(
        device="cpu",
        batch_size=4,
        micro_batch_size=4,
        block_size=32,
        max_iters_per_cycle=2,
        eval_iters=2,
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

    logger = sf.logging.getLogger("swarm_forge_test_smoke")
    dataset = sf.TinyShakespeareData(str(data_dir), train_cfg, logger)
    model_cfg.vocab_size = dataset.tokenizer.vocab_size

    runtime = sf.TrainingRuntime(
        tcfg=train_cfg,
        mcfg=model_cfg,
        dataset=dataset,
        output_dir=tmp_path / "runs",
        logger=logger,
    )

    metrics_before = runtime.evaluate()
    train_stats = runtime.train_steps(1)
    metrics_after = runtime.evaluate()

    assert "val_loss" in metrics_before
    assert "val_loss" in metrics_after
    assert train_stats["global_step"] >= 1
    assert metrics_after["perplexity"] > 0.0

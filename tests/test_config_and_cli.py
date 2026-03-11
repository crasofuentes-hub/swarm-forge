import swarm_forge as sf


def test_arg_parser_defaults():
    parser = sf.build_arg_parser()
    args = parser.parse_args([])
    assert args.max_hours == 24.0
    assert args.cycle_seconds == 900
    assert args.batch_size == 32


def test_swarm_config_validation_accepts_expected_shape(tmp_path):
    cfg = sf.SwarmConfig(
        output_dir=str(tmp_path / "runs"),
        data_dir=str(tmp_path / "data"),
        max_hours=1.0,
        cycle_seconds=900,
    )
    cfg.validate()
    assert (tmp_path / "runs").exists()
    assert (tmp_path / "data").exists()


def test_swarm_config_rejects_wrong_agent_count():
    cfg = sf.SwarmConfig(agents_per_role=9)
    try:
        cfg.validate()
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "agents_per_role" in str(exc)

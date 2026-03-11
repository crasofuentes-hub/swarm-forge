import swarm_forge as sf


def make_patch(patch_id: str, patch_type: str = "hyperparam", role: str = "HyperparamTuner"):
    return sf.Patch(
        id=patch_id,
        author_id=f"{role}-00",
        role=role,
        timestamp=sf.utc_now(),
        patch_type=patch_type,
        payload={"learning_rate": 3e-4},
        summary="test patch",
        cycle_index=1,
    )


def make_vote(patch_id: str, voter_id: str, voter_role: str, approve: bool, score: float, weight: float):
    return sf.PatchVote(
        patch_id=patch_id,
        voter_id=voter_id,
        voter_role=voter_role,
        approve=approve,
        score=score,
        reason="test",
        weight=weight,
    )


def test_conflict_group_mapping():
    p1 = make_patch("p1", "model_arch", "LayerArchitect")
    p2 = make_patch("p2", "hyperparam", "HyperparamTuner")
    p3 = make_patch("p3", "data", "DataAugmentor")
    assert sf.PatchConflictResolver.conflict_group(p1) == "model_arch"
    assert sf.PatchConflictResolver.conflict_group(p2) == "training_config"
    assert sf.PatchConflictResolver.conflict_group(p3) == "data"


def test_weighted_threshold_logic_accepts_high_score_patch(tmp_path):
    engine = object.__new__(sf.SwarmEngine)
    engine.swarm_cfg = sf.SwarmConfig(
        output_dir=str(tmp_path / "runs"),
        data_dir=str(tmp_path / "data"),
        patch_apply_approval_threshold=60.0,
        patch_apply_score_threshold=70.0,
    )

    patch = make_patch("p-good")
    votes = []

    # 10 evaluators -> 50 total weight => 5 each
    for i in range(10):
        votes.append(make_vote("p-good", f"Evaluator-{i:02d}", "Evaluator", True, 82.0, 5.0))

    # 110 non-evaluators -> 50 total weight => 50/110 each
    other_weight = 50.0 / 110.0
    for i in range(110):
        approve = i < 80
        score = 76.0 if approve else 55.0
        votes.append(make_vote("p-good", f"Other-{i:03d}", "BugHunter", approve, score, other_weight))

    decisions = sf.SwarmEngine._decide_patches(engine, [patch], {"p-good": votes})
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.applied is True
    assert decision.approval_percent >= 60.0
    assert decision.weighted_score >= 70.0


def test_weighted_threshold_logic_rejects_low_score_patch(tmp_path):
    engine = object.__new__(sf.SwarmEngine)
    engine.swarm_cfg = sf.SwarmConfig(
        output_dir=str(tmp_path / "runs"),
        data_dir=str(tmp_path / "data"),
        patch_apply_approval_threshold=60.0,
        patch_apply_score_threshold=70.0,
    )

    patch = make_patch("p-bad")
    votes = []

    for i in range(10):
        votes.append(make_vote("p-bad", f"Evaluator-{i:02d}", "Evaluator", True, 62.0, 5.0))

    other_weight = 50.0 / 110.0
    for i in range(110):
        votes.append(make_vote("p-bad", f"Other-{i:03d}", "BugHunter", True, 61.0, other_weight))

    decisions = sf.SwarmEngine._decide_patches(engine, [patch], {"p-bad": votes})
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.applied is False
    assert decision.weighted_score < 70.0

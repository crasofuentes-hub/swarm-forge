from collections import defaultdict
from pathlib import Path

import swarm_forge as sf


class DummyParallel:
    def set_state(self, worker, state):
        return None


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def build_engine_stub(tmp_path: Path):
    engine = object.__new__(sf.SwarmEngine)
    engine.swarm_cfg = sf.SwarmConfig(
        output_dir=str(tmp_path / "runs"),
        data_dir=str(tmp_path / "data"),
        max_hours=1.0,
    )
    engine.lifecycle_path = tmp_path / "lifecycle.jsonl"
    engine.parallel = DummyParallel()
    engine.logger = DummyLogger()
    engine.workers = {}
    engine.role_to_agent_ids = defaultdict(list)
    engine.agent_states = {}
    return engine


def test_agent_dies_after_three_failures(tmp_path):
    engine = build_engine_stub(tmp_path)

    st0 = sf.AgentState(agent_id="BugHunter-00", role="BugHunter", alive=True)
    st1 = sf.AgentState(agent_id="BugHunter-01", role="BugHunter", alive=True)

    engine.agent_states = {
        "BugHunter-00": st0,
        "BugHunter-01": st1,
    }
    engine.role_to_agent_ids["BugHunter"] = ["BugHunter-00", "BugHunter-01"]
    engine.workers = {"BugHunter-00": object(), "BugHunter-01": object()}

    engine._mark_agent_failure("BugHunter-00", "e1")
    assert engine.agent_states["BugHunter-00"].alive is True
    assert engine.agent_states["BugHunter-00"].consecutive_failures == 1

    engine._mark_agent_failure("BugHunter-00", "e2")
    assert engine.agent_states["BugHunter-00"].alive is True
    assert engine.agent_states["BugHunter-00"].consecutive_failures == 2

    engine._mark_agent_failure("BugHunter-00", "e3")
    assert engine.agent_states["BugHunter-00"].alive is False
    assert engine.agent_states["BugHunter-00"].deaths == 1


def test_role_reassignment_increments_survivor_counter(tmp_path):
    engine = build_engine_stub(tmp_path)

    dead = sf.AgentState(agent_id="BugHunter-00", role="BugHunter", alive=False)
    alive = sf.AgentState(agent_id="BugHunter-01", role="BugHunter", alive=True)

    engine.agent_states = {
        "BugHunter-00": dead,
        "BugHunter-01": alive,
    }
    engine.role_to_agent_ids["BugHunter"] = ["BugHunter-00", "BugHunter-01"]
    engine.workers = {"BugHunter-00": object(), "BugHunter-01": object()}

    engine._reassign_role_lottery("BugHunter-00", "BugHunter")
    assert engine.agent_states["BugHunter-01"].reassignments == 1

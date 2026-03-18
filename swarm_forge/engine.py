"""Engine and orchestration surface for Swarm Forge."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict

from .agents import AgentState, ROLE_TO_CLASS, LocalAgentWorker
from .patches import Patch, PatchVote


class ParallelBackend:
    def __init__(self, use_ray: bool, logger: logging.Logger):
        self.use_ray = False
        self.logger = logger
        self.ray = None
        self.remote_worker_cls = None
        if use_ray:
            self._try_init_ray()

    def _try_init_ray(self) -> None:
        try:
            import ray  # type: ignore
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
            self.ray = ray

            @ray.remote
            class RemoteAgentWorker:
                def __init__(self, role_name: str, agent_id: str, seed: int):
                    cls = ROLE_TO_CLASS[role_name]
                    self.agent = cls(agent_id=agent_id, role=role_name, seed=seed)

                def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Dict[str, Any]:
                    patch = self.agent.generate_patch(runtime_state, cycle_index)
                    return asdict(patch)

                def score_patch(self, patch_dict: Dict[str, Any], runtime_metrics: Dict[str, Any], runtime_state: Dict[str, Any]) -> Dict[str, Any]:
                    patch = Patch(**patch_dict)
                    vote = self.agent.score_patch(patch, runtime_metrics, runtime_state)
                    return asdict(vote)

                def get_state(self) -> Dict[str, Any]:
                    return asdict(self.agent.state)

                def set_state(self, state_dict: Dict[str, Any]) -> None:
                    self.agent.state = AgentState(**state_dict)

            self.remote_worker_cls = RemoteAgentWorker
            self.use_ray = True
            self.logger.info("Parallel backend: Ray enabled.")
        except Exception as exc:
            self.logger.warning("Ray unavailable; using local fallback. reason=%s", exc)
            self.use_ray = False

    def create_worker(self, role_name: str, agent_id: str, seed: int) -> Any:
        if self.use_ray:
            return self.remote_worker_cls.remote(role_name=role_name, agent_id=agent_id, seed=seed)
        cls = ROLE_TO_CLASS[role_name]
        return LocalAgentWorker(cls(agent_id=agent_id, role=role_name, seed=seed))

    def generate_patch(self, worker: Any, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        if self.use_ray:
            result = self.ray.get(worker.generate_patch.remote(runtime_state, cycle_index))
            return Patch(**result)
        return worker.generate_patch(runtime_state, cycle_index)

    def score_patch(self, worker: Any, patch: Patch, runtime_metrics: Dict[str, Any], runtime_state: Dict[str, Any]) -> PatchVote:
        if self.use_ray:
            result = self.ray.get(worker.score_patch.remote(asdict(patch), runtime_metrics, runtime_state))
            return PatchVote(**result)
        return worker.score_patch(patch, runtime_metrics, runtime_state)

    def get_state(self, worker: Any) -> AgentState:
        if self.use_ray:
            result = self.ray.get(worker.get_state.remote())
            return AgentState(**result)
        return worker.get_state()

    def set_state(self, worker: Any, state: AgentState) -> None:
        if self.use_ray:
            self.ray.get(worker.set_state.remote(asdict(state)))
        else:
            worker.agent.state = state

    def shutdown(self) -> None:
        if self.use_ray and self.ray is not None:
            try:
                self.ray.shutdown()
            except Exception:
                pass


def __getattr__(name: str):
    if name in {"TrainingRuntime", "SwarmEngine"}:
        from . import core as _core
        return getattr(_core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TrainingRuntime",
    "ParallelBackend",
    "SwarmEngine",
]
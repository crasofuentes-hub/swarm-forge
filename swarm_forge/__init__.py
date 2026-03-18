"""Swarm Forge package public API."""

import logging

from .config import TrainingConfig, ModelConfig, SwarmConfig
from .core import build_arg_parser, main, utc_now
from .data import CharTokenizer, TinyShakespeareData, WikiText2Data, build_dataset
from .engine import TrainingRuntime, ParallelBackend, SwarmEngine
from .patches import Patch, PatchVote, PatchDecision, PatchConflictResolver
from .agents import (
    AgentState,
    Agent,
    BugHunterAgent,
    HyperparamTunerAgent,
    LayerArchitectAgent,
    TokenizerOptimizerAgent,
    DataAugmentorAgent,
    LossEngineerAgent,
    SaboteurAgent,
    ResilienceCheckerAgent,
    SpeedDemonAgent,
    MemoryWardenAgent,
    EvaluatorAgent,
    ArbiterAgent,
    build_patch,
    ROLE_TO_CLASS,
    LocalAgentWorker,
)

__all__ = [
    "TrainingConfig",
    "ModelConfig",
    "SwarmConfig",
    "CharTokenizer",
    "TinyShakespeareData",
    "WikiText2Data",
    "build_dataset",
    "TrainingRuntime",
    "ParallelBackend",
    "SwarmEngine",
    "Patch",
    "PatchVote",
    "PatchDecision",
    "PatchConflictResolver",
    "AgentState",
    "Agent",
    "BugHunterAgent",
    "HyperparamTunerAgent",
    "LayerArchitectAgent",
    "TokenizerOptimizerAgent",
    "DataAugmentorAgent",
    "LossEngineerAgent",
    "SaboteurAgent",
    "ResilienceCheckerAgent",
    "SpeedDemonAgent",
    "MemoryWardenAgent",
    "EvaluatorAgent",
    "ArbiterAgent",
    "build_patch",
    "ROLE_TO_CLASS",
    "LocalAgentWorker",
    "build_arg_parser",
    "main",
    "utc_now",
    "logging",
]
"""Swarm Forge package public API."""

import logging
from dataclasses import asdict

from .config import TrainingConfig, ModelConfig, SwarmConfig
from .core import TrainingRuntime, SwarmEngine, build_arg_parser, main, utc_now
from .data import CharTokenizer, TinyShakespeareData, WikiText2Data, build_dataset
from .engine import ParallelBackend
from .patches import Patch, PatchVote, PatchDecision, PatchConflictResolver
from .proposals import AppliedPatchRecord, ExperimentProposal, build_experiment_proposal
from .patch_applier import PatchApplier
from .research import CampaignConfig, TrialSpec, TrialResult, CampaignSummary, CampaignRunner, score_trial_result, select_best_trial, build_campaign_summary, proposal_to_trial_spec, proposals_to_trial_specs
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
    "AppliedPatchRecord",
    "ExperimentProposal",
    "build_experiment_proposal",
    "PatchApplier",
    "CampaignConfig",
    "TrialSpec",
    "TrialResult",
    "CampaignSummary",
    "CampaignRunner",
    "score_trial_result",
    "select_best_trial",
    "build_campaign_summary",
    "proposal_to_trial_spec",
    "proposals_to_trial_specs",
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
    "asdict",
]
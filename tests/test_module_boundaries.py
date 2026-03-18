from swarm_forge.config import TrainingConfig, ModelConfig, SwarmConfig
from swarm_forge.data import CharTokenizer, TinyShakespeareData, WikiText2Data, build_dataset
from swarm_forge.proposals import AppliedPatchRecord, ExperimentProposal, build_experiment_proposal
from swarm_forge.patches import Patch, PatchVote, PatchDecision, PatchConflictResolver
from swarm_forge.core import PatchApplier, build_patch
from swarm_forge.agents import (
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
    LocalAgentWorker,
)
from swarm_forge.engine import TrainingRuntime, ParallelBackend, SwarmEngine


def test_modular_public_surfaces_are_importable() -> None:
    assert TrainingConfig is not None
    assert ModelConfig is not None
    assert SwarmConfig is not None

    assert CharTokenizer is not None
    assert TinyShakespeareData is not None
    assert WikiText2Data is not None
    assert build_dataset is not None

    assert AppliedPatchRecord is not None
    assert ExperimentProposal is not None
    assert build_experiment_proposal is not None

    assert Patch is not None
    assert PatchVote is not None
    assert PatchDecision is not None
    assert PatchConflictResolver is not None
    assert PatchApplier is not None
    assert build_patch is not None

    assert AgentState is not None
    assert Agent is not None
    assert BugHunterAgent is not None
    assert HyperparamTunerAgent is not None
    assert LayerArchitectAgent is not None
    assert TokenizerOptimizerAgent is not None
    assert DataAugmentorAgent is not None
    assert LossEngineerAgent is not None
    assert SaboteurAgent is not None
    assert ResilienceCheckerAgent is not None
    assert SpeedDemonAgent is not None
    assert MemoryWardenAgent is not None
    assert EvaluatorAgent is not None
    assert ArbiterAgent is not None
    assert LocalAgentWorker is not None

    assert TrainingRuntime is not None
    assert ParallelBackend is not None
    assert SwarmEngine is not None
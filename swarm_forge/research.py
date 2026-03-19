"""Research orchestration contracts and scoring helpers for Swarm Forge."""

from __future__ import annotations

from pathlib import Path
import logging

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .proposals import ExperimentProposal
from .config import TrainingConfig, ModelConfig
from .data import build_dataset
from .core import TrainingRuntime
from .common import ensure_dir


@dataclass
class CampaignConfig:
    campaign_id: str
    dataset_name: str
    objective_metric: str = "val_loss"
    maximize: bool = False
    notes: str = ""


@dataclass
class TrialSpec:
    trial_id: str
    campaign_id: str
    hypothesis: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class TrialResult:
    trial_id: str
    campaign_id: str
    success: bool
    objective_metric: str
    objective_value: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    notes: str = ""


@dataclass
class CampaignSummary:
    campaign_id: str
    objective_metric: str
    best_trial_id: Optional[str]
    best_objective_value: Optional[float]
    total_trials: int
    successful_trials: int


def score_trial_result(result: TrialResult, maximize: bool = False) -> float:
    value = float(result.objective_value)
    return value if maximize else -value


def select_best_trial(results: List[TrialResult], maximize: bool = False) -> Optional[TrialResult]:
    successful = [r for r in results if r.success]
    if not successful:
        return None
    return max(successful, key=lambda r: score_trial_result(r, maximize=maximize))


def build_campaign_summary(
    campaign_id: str,
    objective_metric: str,
    results: List[TrialResult],
    maximize: bool = False,
) -> CampaignSummary:
    best = select_best_trial(results, maximize=maximize)
    successful_trials = sum(1 for r in results if r.success)
    return CampaignSummary(
        campaign_id=campaign_id,
        objective_metric=objective_metric,
        best_trial_id=best.trial_id if best is not None else None,
        best_objective_value=best.objective_value if best is not None else None,
        total_trials=len(results),
        successful_trials=successful_trials,
    )
def proposal_to_trial_spec(proposal: ExperimentProposal, campaign_id: Optional[str] = None) -> TrialSpec:
    resolved_campaign_id = campaign_id or f"{proposal.dataset_name}:{proposal.success_metric}"
    overrides = {
        proposal.changed_variable: proposal.proposed_value,
    }
    tags = [
        proposal.dataset_name,
        proposal.author_role,
        proposal.changed_variable,
    ]
    return TrialSpec(
        trial_id=proposal.proposal_id,
        campaign_id=resolved_campaign_id,
        hypothesis=proposal.hypothesis,
        overrides=overrides,
        tags=tags,
    )


def proposals_to_trial_specs(
    proposals: List[ExperimentProposal],
    campaign_id: Optional[str] = None,
) -> List[TrialSpec]:
    return [proposal_to_trial_spec(p, campaign_id=campaign_id) for p in proposals]
class CampaignRunner:
    def __init__(self, config: CampaignConfig):
        self.config = config
        self.trials: List[TrialSpec] = []
        self.results: List[TrialResult] = []

    def add_trial(self, trial: TrialSpec) -> None:
        if trial.campaign_id != self.config.campaign_id:
            raise ValueError("Trial campaign_id does not match CampaignRunner config.")
        self.trials.append(trial)

    def add_trials(self, trials: List[TrialSpec]) -> None:
        for trial in trials:
            self.add_trial(trial)

    def add_result(self, result: TrialResult) -> None:
        if result.campaign_id != self.config.campaign_id:
            raise ValueError("TrialResult campaign_id does not match CampaignRunner config.")
        self.results.append(result)

    def add_results(self, results: List[TrialResult]) -> None:
        for result in results:
            self.add_result(result)

    def run_trial(self, executor: "TrialExecutor", trial: TrialSpec) -> TrialResult:
        self.add_trial(trial)
        result = executor.execute(trial)
        self.add_result(result)
        return result

    def run_trials(self, executor: "TrialExecutor", trials: List[TrialSpec]) -> List[TrialResult]:
        out: List[TrialResult] = []
        for trial in trials:
            out.append(self.run_trial(executor, trial))
        return out

    def rank_results(self) -> List[TrialResult]:
        successful = [r for r in self.results if r.success]
        return sorted(
            successful,
            key=lambda r: (
                score_trial_result(r, maximize=self.config.maximize),
                r.trial_id,
            ),
            reverse=True,
        )

    def best_result(self) -> Optional[TrialResult]:
        ranked = self.rank_results()
        return ranked[0] if ranked else None

    def best_trial_id(self) -> Optional[str]:
        best = self.best_result()
        return best.trial_id if best is not None else None

    def summary(self) -> CampaignSummary:
        return build_campaign_summary(
            campaign_id=self.config.campaign_id,
            objective_metric=self.config.objective_metric,
            results=self.results,
            maximize=self.config.maximize,
        )
class TrialExecutor:
    def __init__(
        self,
        campaign: CampaignConfig,
        base_train_cfg: TrainingConfig,
        base_model_cfg: ModelConfig,
        data_dir: str,
        output_root: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.campaign = campaign
        self.base_train_cfg = base_train_cfg
        self.base_model_cfg = base_model_cfg
        self.data_dir = data_dir
        self.output_root = output_root
        self.logger = logger or logging.getLogger("swarm_forge_trial_executor")

    def _build_trial_train_cfg(self, trial: TrialSpec) -> TrainingConfig:
        payload = asdict(self.base_train_cfg)
        for key, value in trial.overrides.items():
            if key in payload:
                payload[key] = value
        return TrainingConfig(**payload)

    def _build_trial_model_cfg(self) -> ModelConfig:
        return ModelConfig(**asdict(self.base_model_cfg))

    def execute(self, trial: TrialSpec) -> TrialResult:
        train_cfg = self._build_trial_train_cfg(trial)
        model_cfg = self._build_trial_model_cfg()

        trial_output_dir = ensure_dir(Path(self.output_root) / self.campaign.campaign_id / trial.trial_id)
        dataset = build_dataset(self.campaign.dataset_name, self.data_dir, train_cfg, self.logger)
        model_cfg.vocab_size = dataset.tokenizer.vocab_size
        model_cfg.block_size = train_cfg.block_size

        runtime = TrainingRuntime(
            tcfg=train_cfg,
            mcfg=model_cfg,
            dataset=dataset,
            output_dir=trial_output_dir,
            logger=self.logger,
        )

        initial_metrics = runtime.evaluate()
        train_stats = runtime.train_steps(train_cfg.patch_trial_train_steps)
        final_metrics = runtime.evaluate()

        objective_metric = self.campaign.objective_metric
        objective_value = float(final_metrics[objective_metric])

        checkpoint_path = runtime.save_checkpoint("trial_final", final_metrics)

        metrics = {
            "initial": initial_metrics,
            "train": asdict(train_stats),
            "final": final_metrics,
        }

        return TrialResult(
            trial_id=trial.trial_id,
            campaign_id=trial.campaign_id,
            success=True,
            objective_metric=objective_metric,
            objective_value=objective_value,
            metrics=metrics,
            checkpoint_path=str(checkpoint_path),
            notes=trial.hypothesis,
        )
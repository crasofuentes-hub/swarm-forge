"""Core contracts for Swarm Forge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RuntimeSnapshot:
    training_config: Dict[str, Any]
    model_config: Dict[str, Any]
    loss_name: str
    best_val_loss: float
    global_step: int
    tokenizer_merges: list
    dataset_text_hash: str
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Optional[Dict[str, Any]]
@dataclass
class ExecutionResult:
    train_loss_recent: float
    train_throughput_tokens_per_sec: float
    global_step: int

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def items(self):
        return {
            "train_loss_recent": self.train_loss_recent,
            "train_throughput_tokens_per_sec": self.train_throughput_tokens_per_sec,
            "global_step": self.global_step,
        }.items()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_loss_recent": self.train_loss_recent,
            "train_throughput_tokens_per_sec": self.train_throughput_tokens_per_sec,
            "global_step": self.global_step,
        }
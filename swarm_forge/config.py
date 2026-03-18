"""Configuration models for Swarm Forge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError as exc:
    raise RuntimeError("PyTorch is required. Install torch before running this script.") from exc


@dataclass
class TrainingConfig:
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    compile_model: bool = False
    learning_rate: float = 3e-4
    min_learning_rate: float = 5e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    batch_size: int = 32
    micro_batch_size: int = 32
    block_size: int = 256
    max_iters_per_cycle: int = 800
    eval_iters: int = 40
    warmup_iters: int = 100
    dropout: float = 0.2
    amp_enabled: bool = True
    use_gradient_checkpointing: bool = False
    pin_memory: bool = True
    num_workers: int = 2
    persistent_workers: bool = False
    prefetch_factor: int = 2
    label_smoothing: float = 0.0
    focal_gamma: float = 0.0
    scheduler_name: str = "cosine"
    max_augmented_fraction: float = 0.10
    train_val_split: float = 0.90
    vocab_mode: str = "char"
    stress_noise_std: float = 0.0
    sabotage_eval_only: bool = True
    max_sabotage_intensity: float = 0.25
    quantize_eval_only: bool = False
    torch_compile_backend: str = "inductor"
    gradient_accumulation_steps: int = 1
    checkpoint_keep_last: int = 3
    objective_metric: str = "val_loss"
    focused_search: bool = True
    patch_trial_train_steps: int = 40


@dataclass
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True
    use_gru_gate: bool = False
    residual_scale: float = 1.0
    rope_like_scale: float = 1.0


@dataclass
class SwarmConfig:
    agents_per_role: int = 10
    total_roles: int = 12
    cycle_seconds: int = 900
    max_hours: float = 24.0
    max_cycles: Optional[int] = None
    output_dir: str = "runs/swarm_forge_v1"
    data_dir: str = "data/tinyshakespeare"
    dataset_name: str = "tinyshakespeare"
    use_ray: bool = True
    fallback_workers: int = 16
    patch_apply_approval_threshold: float = 60.0
    patch_apply_score_threshold: float = 70.0
    max_patches_applied_per_cycle: int = 6
    max_arch_delta_layers: int = 2
    max_arch_delta_embd: int = 128
    max_arch_delta_heads: int = 2
    evaluator_weight_total: float = 50.0
    others_weight_total: float = 50.0
    dead_after_failures: int = 3
    max_patch_eval_candidates: Optional[int] = None
    quiet_sleep: float = 1.0
    reduced_roles_mode: bool = True

    def validate(self) -> None:
        if self.agents_per_role != 10:
            raise ValueError("agents_per_role must be exactly 10.")
        if self.total_roles != 12:
            raise ValueError("total_roles must be exactly 12.")
        if self.cycle_seconds <= 0:
            raise ValueError("cycle_seconds must be positive.")
        if self.max_hours <= 0 and self.max_cycles is None:
            raise ValueError("Either max_hours > 0 or max_cycles must be provided.")
        if self.patch_apply_approval_threshold < 0 or self.patch_apply_approval_threshold > 100:
            raise ValueError("Invalid approval threshold.")
        if self.patch_apply_score_threshold < 0 or self.patch_apply_score_threshold > 100:
            raise ValueError("Invalid score threshold.")
        if abs(self.evaluator_weight_total + self.others_weight_total - 100.0) > 1e-9:
            raise ValueError("Voting weights must sum to 100.")
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)


__all__ = [
    "TrainingConfig",
    "ModelConfig",
    "SwarmConfig",
]
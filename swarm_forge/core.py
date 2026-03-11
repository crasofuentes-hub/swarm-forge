# swarm_forge.py
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import signal
import sys
import time
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
except ImportError as exc:
    raise RuntimeError("PyTorch is required. Install torch before running this script.") from exc

__version__ = "1.0.0"

ROLE_NAMES: Tuple[str, ...] = (
    "BugHunter",
    "HyperparamTuner",
    "LayerArchitect",
    "TokenizerOptimizer",
    "DataAugmentor",
    "LossEngineer",
    "Saboteur",
    "ResilienceChecker",
    "SpeedDemon",
    "MemoryWarden",
    "Evaluator",
    "Arbiter",
)

PATCH_TYPES: Tuple[str, ...] = (
    "bugfix",
    "hyperparam",
    "model_arch",
    "tokenizer",
    "data",
    "loss",
    "sabotage",
    "resilience",
    "speed",
    "memory",
    "evaluation",
    "arbitration",
)

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(data, sort_keys=True, default=str) + "\n")


def current_memory_allocated(device: str) -> int:
    if device.startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.memory_allocated())
    return 0


def current_max_memory_allocated(device: str) -> int:
    if device.startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0


def human_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def count_alive_by_role(agent_states: Sequence["AgentState"]) -> Dict[str, int]:
    c: Dict[str, int] = Counter()
    for st in agent_states:
        if st.alive:
            c[st.role] += 1
    return dict(sorted(c.items(), key=lambda kv: kv[0]))


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.merges: List[Tuple[str, str]] = []

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s if ch in self.stoi]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids if int(i) in self.itos)

    def update_merges(self, merges: List[Tuple[str, str]]) -> None:
        self.merges = list(merges)


class TinyShakespeareData:
    def __init__(self, data_dir: str, cfg: TrainingConfig, logger: logging.Logger):
        self.data_dir = ensure_dir(data_dir)
        self.cfg = cfg
        self.logger = logger
        self.raw_path = self.data_dir / "input.txt"
        self.text = self._prepare_dataset()
        self.tokenizer = CharTokenizer(self.text)
        self.train_ids, self.val_ids = self._build_ids(self.text)
        self.augmentation_history: List[Dict[str, Any]] = []

    def _prepare_dataset(self) -> str:
        if not self.raw_path.exists():
            self.logger.info("Downloading TinyShakespeare dataset to %s", self.raw_path)
            urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, self.raw_path)
        text = self.raw_path.read_text(encoding="utf-8")
        if len(text) < 10000:
            raise ValueError("TinyShakespeare dataset appears incomplete or corrupted.")
        return text

    def _build_ids(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        split = int(len(encoded) * self.cfg.train_val_split)
        return encoded[:split], encoded[split:]

    def rebuild_after_tokenizer_update(self) -> None:
        self.train_ids, self.val_ids = self._build_ids(self.text)

    def apply_augmentation(self, intensity: float, pattern: str) -> Dict[str, Any]:
        intensity = max(0.0, min(float(intensity), self.cfg.max_augmented_fraction))
        n_chars = int(len(self.text) * intensity)
        if n_chars == 0:
            return {"changed_chars": 0, "pattern": pattern}
        rng = random.Random(self.cfg.seed + len(self.augmentation_history))
        text_list = list(self.text)
        indices = rng.sample(range(len(text_list)), k=min(n_chars, len(text_list)))
        substitutions = {
            "a": "A",
            "e": "E",
            "i": "I",
            "o": "O",
            "u": "U",
            ",": ";",
            ".": "!",
            "\n": "\n",
        }
        for idx in indices:
            ch = text_list[idx]
            if pattern == "case_flip":
                text_list[idx] = ch.swapcase()
            elif pattern == "punctuation":
                text_list[idx] = substitutions.get(ch, ch)
            elif pattern == "space_noise":
                text_list[idx] = " " if ch.isalpha() and rng.random() < 0.1 else ch
            else:
                text_list[idx] = ch
        self.text = "".join(text_list)
        self.augmentation_history.append({"ts": utc_now(), "changed_chars": len(indices), "pattern": pattern})
        self.rebuild_after_tokenizer_update()
        return {"changed_chars": len(indices), "pattern": pattern}

    def get_batch(self, split: Literal["train", "val"], batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_ids if split == "train" else self.val_ids
        if len(data) <= block_size + 1:
            raise ValueError("Dataset is too small for the configured block size.")
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
        if device.startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


@dataclass
class AppliedPatchRecord:
    patch_id: str
    patch_type: str
    author_id: str
    role: str
    applied_at: str
    payload_hash: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size)
        self.register_buffer("bias_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)
        self.use_gru_gate = cfg.use_gru_gate
        self.residual_scale = cfg.residual_scale
        if self.use_gru_gate:
            self.gru_gate = nn.GRUCell(cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.ln_1(x))
        x = x + self.residual_scale * h
        m = self.mlp(self.ln_2(x))
        if self.use_gru_gate:
            B, T, C = x.shape
            flat_x = x.reshape(B * T, C)
            flat_m = m.reshape(B * T, C)
            g = self.gru_gate(flat_m, flat_x).reshape(B, T, C)
            x = x + self.residual_scale * g
        else:
            x = x + self.residual_scale * m
        return x


class NanoGPT(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(model_cfg.vocab_size, model_cfg.n_embd),
            wpe=nn.Embedding(model_cfg.block_size, model_cfg.n_embd),
            drop=nn.Dropout(model_cfg.dropout),
            h=nn.ModuleList([Block(model_cfg) for _ in range(model_cfg.n_layer)]),
            ln_f=LayerNorm(model_cfg.n_embd, bias=model_cfg.bias),
        ))
        self.lm_head = nn.Linear(model_cfg.n_embd, model_cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * model_cfg.n_layer))
        self.gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def crop_block_size(self, block_size: int) -> None:
        self.model_cfg.block_size = block_size
        self.transformer.wpe = nn.Embedding(block_size, self.model_cfg.n_embd)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, loss_fn: Optional[Callable[..., torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        if T > self.model_cfg.block_size:
            raise ValueError(f"Cannot forward sequence of length {T}; block size is only {self.model_cfg.block_size}")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos) * self.model_cfg.rope_like_scale
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            if loss_fn is None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
class TrainingRuntime:
    def __init__(self, tcfg: TrainingConfig, mcfg: ModelConfig, dataset: TinyShakespeareData, output_dir: Path, logger: logging.Logger):
        self.tcfg = tcfg
        self.mcfg = mcfg
        self.dataset = dataset
        self.output_dir = output_dir
        self.logger = logger
        self.device = tcfg.device
        self.model = NanoGPT(mcfg).to(self.device)
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.applied_patches: List[AppliedPatchRecord] = []
        self.loss_name: str = "cross_entropy"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_fp16_scaler())
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self._maybe_compile()
        self.last_eval_metrics: Dict[str, Any] = {}
        self.state_backups_dir = ensure_dir(self.output_dir / "state_backups")
        self.ckpt_dir = ensure_dir(self.output_dir / "checkpoints")

    def _use_fp16_scaler(self) -> bool:
        return self.device.startswith("cuda") and self.tcfg.amp_enabled and self.tcfg.dtype == "float16"

    def _amp_dtype(self) -> torch.dtype:
        if self.tcfg.dtype == "bfloat16":
            return torch.bfloat16
        if self.tcfg.dtype == "float16":
            return torch.float16
        return torch.float32

    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay_params = []
        no_decay_params = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)
        optim_groups = [
            {"params": decay_params, "weight_decay": self.tcfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        fused_available = self.device.startswith("cuda") and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        extra = {"fused": True} if fused_available else {}
        return torch.optim.AdamW(
            optim_groups,
            lr=self.tcfg.learning_rate,
            betas=(self.tcfg.beta1, self.tcfg.beta2),
            **extra,
        )

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if self.tcfg.scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.tcfg.max_iters_per_cycle),
                eta_min=self.tcfg.min_learning_rate,
            )
        if self.tcfg.scheduler_name == "none":
            return None
        if self.tcfg.scheduler_name == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.2,
                total_iters=max(1, self.tcfg.max_iters_per_cycle),
            )
        return None

    def _maybe_compile(self) -> None:
        if self.tcfg.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, backend=self.tcfg.torch_compile_backend)
                self.logger.info("Model compiled with torch.compile backend=%s", self.tcfg.torch_compile_backend)
            except Exception as exc:
                self.logger.warning("torch.compile failed, continuing without compile: %s", exc)

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "training_config": asdict(self.tcfg),
            "model_config": asdict(self.mcfg),
            "loss_name": self.loss_name,
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "tokenizer_merges": list(self.dataset.tokenizer.merges),
            "dataset_text_hash": stable_hash(self.dataset.text[:50000]),
            "model_state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
        }

    def restore_state(self, snapshot: Dict[str, Any]) -> None:
        self.tcfg = TrainingConfig(**snapshot["training_config"])
        self.mcfg = ModelConfig(**snapshot["model_config"])
        self.loss_name = snapshot["loss_name"]
        self.best_val_loss = snapshot["best_val_loss"]
        self.global_step = snapshot["global_step"]
        self.dataset.tokenizer.update_merges(snapshot["tokenizer_merges"])
        self.model = NanoGPT(self.mcfg).to(self.device)
        self.model.load_state_dict(snapshot["model_state_dict"])
        self.model.gradient_checkpointing = self.tcfg.use_gradient_checkpointing
        self.optimizer = self._build_optimizer()
        self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        self.scheduler = self._build_scheduler()
        if self.scheduler is not None and snapshot["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(snapshot["scheduler_state_dict"])

    def active_loss_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(logits, targets, label_smoothing=self.tcfg.label_smoothing)

        def focal(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.tcfg.label_smoothing)
            pt = torch.exp(-ce)
            gamma = max(0.0, self.tcfg.focal_gamma)
            loss = ((1 - pt) ** gamma) * ce
            return loss.mean()

        if self.loss_name == "focal":
            return focal
        return cross_entropy

    def save_checkpoint(self, tag: str, metrics: Dict[str, Any]) -> Path:
        ckpt = {
            "timestamp": utc_now(),
            "training_config": asdict(self.tcfg),
            "model_config": asdict(self.mcfg),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "last_eval_metrics": metrics,
            "loss_name": self.loss_name,
            "tokenizer_merges": list(self.dataset.tokenizer.merges),
        }
        path = self.ckpt_dir / f"checkpoint_{tag}.pt"
        torch.save(ckpt, path)
        self._cleanup_old_checkpoints()
        return path

    def _cleanup_old_checkpoints(self) -> None:
        files = sorted(self.ckpt_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for extra in files[self.tcfg.checkpoint_keep_last:]:
            try:
                extra.unlink()
            except OSError:
                pass

    def estimate_bleu_like(self, samples: List[str], refs: List[str]) -> float:
        if not samples or not refs:
            return 0.0
        total_score = 0.0
        for hyp, ref in zip(samples, refs):
            hyp_tokens = hyp.split()
            ref_tokens = ref.split()
            if not hyp_tokens or not ref_tokens:
                continue
            overlap = sum(1 for tok in hyp_tokens if tok in ref_tokens)
            total_score += overlap / max(1, len(hyp_tokens))
        return 100.0 * (total_score / max(1, len(samples)))

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        self.model.eval()
        losses = {}
        start = time.perf_counter()
        loss_fn = self.active_loss_fn()
        for split in ("train", "val"):
            out = torch.zeros(self.tcfg.eval_iters)
            for k in range(self.tcfg.eval_iters):
                X, Y = self.dataset.get_batch(split, self.tcfg.batch_size, self.tcfg.block_size, self.device)
                autocast_enabled = self.tcfg.amp_enabled and self.device.startswith("cuda")
                with torch.autocast(
                    device_type="cuda" if self.device.startswith("cuda") else "cpu",
                    dtype=self._amp_dtype(),
                    enabled=autocast_enabled,
                ):
                    _, loss = self.model(X, Y, loss_fn=loss_fn)
                out[k] = loss.detach().float().cpu()
            losses[split] = out.mean().item()
        elapsed = max(1e-9, time.perf_counter() - start)
        tokens = self.tcfg.eval_iters * self.tcfg.batch_size * self.tcfg.block_size * 2
        throughput = tokens / elapsed
        samples = []
        refs = []
        for _ in range(3):
            X, Y = self.dataset.get_batch("val", 1, min(64, self.tcfg.block_size), self.device)
            logits, _ = self.model(X[:, :-1], None, loss_fn=loss_fn)
            pred = logits.argmax(dim=-1)[0].detach().cpu().tolist()
            tgt = Y[0, :-1].detach().cpu().tolist()
            samples.append(self.dataset.tokenizer.decode(pred))
            refs.append(self.dataset.tokenizer.decode(tgt))
        bleu_like = self.estimate_bleu_like(samples, refs)
        metrics = {
            "train_loss": float(losses["train"]),
            "val_loss": float(losses["val"]),
            "perplexity": float(math.exp(min(20.0, losses["val"]))),
            "bleu_like": float(bleu_like),
            "throughput_tokens_per_sec": float(throughput),
            "memory_allocated": current_memory_allocated(self.device),
            "max_memory_allocated": current_max_memory_allocated(self.device),
        }
        self.last_eval_metrics = metrics
        self.model.train()
        return metrics

    def train_steps(self, num_steps: int) -> Dict[str, Any]:
        self.model.train()
        self.model.gradient_checkpointing = self.tcfg.use_gradient_checkpointing
        loss_fn = self.active_loss_fn()
        total_loss = 0.0
        start = time.perf_counter()
        autocast_enabled = self.tcfg.amp_enabled and self.device.startswith("cuda")
        for step in range(num_steps):
            self.optimizer.zero_grad(set_to_none=True)
            X, Y = self.dataset.get_batch("train", self.tcfg.batch_size, self.tcfg.block_size, self.device)
            with torch.autocast(
                device_type="cuda" if self.device.startswith("cuda") else "cpu",
                dtype=self._amp_dtype(),
                enabled=autocast_enabled,
            ):
                _, loss = self.model(X, Y, loss_fn=loss_fn)
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("Training loss became NaN or Inf.")
            total_loss += loss.item()
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1
        elapsed = max(1e-9, time.perf_counter() - start)
        throughput = (num_steps * self.tcfg.batch_size * self.tcfg.block_size) / elapsed
        return {
            "train_loss_recent": total_loss / max(1, num_steps),
            "train_throughput_tokens_per_sec": float(throughput),
            "global_step": self.global_step,
        }


@dataclass
class Patch:
    id: str
    author_id: str
    role: str
    timestamp: str
    patch_type: str
    payload: Dict[str, Any]
    summary: str
    cycle_index: int

    def payload_hash(self) -> str:
        return stable_hash(self.payload)


@dataclass
class PatchVote:
    patch_id: str
    voter_id: str
    voter_role: str
    approve: bool
    score: float
    reason: str
    weight: float = 0.0


@dataclass
class PatchDecision:
    patch_id: str
    approval_percent: float
    weighted_score: float
    applied: bool
    reason: str
    conflict_group: str
    summary: str
    author_id: str
    role: str
    patch_type: str


class PatchConflictResolver:
    @staticmethod
    def conflict_group(patch: Patch) -> str:
        if patch.patch_type in {"model_arch", "tokenizer"}:
            return patch.patch_type
        if patch.patch_type in {"hyperparam", "loss", "memory", "speed"}:
            return "training_config"
        if patch.patch_type in {"sabotage", "resilience"}:
            return "robustness"
        if patch.patch_type == "data":
            return "data"
        return f"misc:{patch.patch_type}"


class PatchApplier:
    def __init__(self, runtime: TrainingRuntime, logger: logging.Logger, swarm_cfg: SwarmConfig):
        self.runtime = runtime
        self.logger = logger
        self.swarm_cfg = swarm_cfg

    def validate_patch(self, patch: Patch) -> None:
        if patch.patch_type not in PATCH_TYPES:
            raise ValueError(f"Unknown patch type: {patch.patch_type}")
        if not isinstance(patch.payload, dict):
            raise ValueError("Patch payload must be a dict.")
        if not patch.summary.strip():
            raise ValueError("Patch summary must not be empty.")

    def apply_patch(self, patch: Patch) -> AppliedPatchRecord:
        self.validate_patch(patch)
        snapshot = self.runtime.snapshot_state()
        try:
            details = self._apply_impl(patch)
            record = AppliedPatchRecord(
                patch_id=patch.id,
                patch_type=patch.patch_type,
                author_id=patch.author_id,
                role=patch.role,
                applied_at=utc_now(),
                payload_hash=patch.payload_hash(),
                success=True,
                details=details,
            )
            self.runtime.applied_patches.append(record)
            return record
        except Exception as exc:
            self.runtime.restore_state(snapshot)
            self.logger.error("Patch application failed; reverted patch=%s error=%s", patch.id, exc)
            return AppliedPatchRecord(
                patch_id=patch.id,
                patch_type=patch.patch_type,
                author_id=patch.author_id,
                role=patch.role,
                applied_at=utc_now(),
                payload_hash=patch.payload_hash(),
                success=False,
                details={"error": str(exc)},
            )

    def _apply_impl(self, patch: Patch) -> Dict[str, Any]:
        p = patch.payload
        tcfg = self.runtime.tcfg
        mcfg = self.runtime.mcfg

        if patch.patch_type == "bugfix":
            if "pin_memory" in p:
                tcfg.pin_memory = bool(p["pin_memory"])
            if "num_workers" in p:
                tcfg.num_workers = max(0, min(8, int(p["num_workers"])))
            if "persistent_workers" in p:
                tcfg.persistent_workers = bool(p["persistent_workers"]) and tcfg.num_workers > 0
            if "prefetch_factor" in p and tcfg.num_workers > 0:
                tcfg.prefetch_factor = max(2, min(8, int(p["prefetch_factor"])))
            return {"updated": "dataloader_runtime_flags"}

        if patch.patch_type == "hyperparam":
            for k in ("learning_rate", "min_learning_rate", "weight_decay", "dropout", "label_smoothing", "focal_gamma"):
                if k in p:
                    setattr(tcfg, k, float(p[k]))
            for k in ("batch_size", "micro_batch_size", "warmup_iters", "gradient_accumulation_steps"):
                if k in p:
                    setattr(tcfg, k, max(1, int(p[k])))
            if "scheduler_name" in p:
                tcfg.scheduler_name = str(p["scheduler_name"])
            self.runtime.optimizer = self.runtime._build_optimizer()
            self.runtime.scheduler = self.runtime._build_scheduler()
            return {"updated": "optimizer_scheduler_hparams"}

        if patch.patch_type == "model_arch":
            new_cfg = ModelConfig(**asdict(mcfg))
            if "n_layer_delta" in p:
                new_cfg.n_layer = max(2, min(16, new_cfg.n_layer + int(p["n_layer_delta"])))
            if "n_head_delta" in p:
                new_cfg.n_head = max(2, min(12, new_cfg.n_head + int(p["n_head_delta"])))
            if "n_embd_delta" in p:
                new_cfg.n_embd = max(128, min(1024, new_cfg.n_embd + int(p["n_embd_delta"])))
                rem = new_cfg.n_embd % new_cfg.n_head
                if rem != 0:
                    new_cfg.n_embd += (new_cfg.n_head - rem)
            if "use_gru_gate" in p:
                new_cfg.use_gru_gate = bool(p["use_gru_gate"])
            if "residual_scale" in p:
                new_cfg.residual_scale = max(0.5, min(1.5, float(p["residual_scale"])))
            new_model = NanoGPT(new_cfg).to(self.runtime.device)
            old_sd = self.runtime.model.state_dict()
            compatible_sd = {}
            new_sd = new_model.state_dict()
            for key, tensor in old_sd.items():
                if key in new_sd and new_sd[key].shape == tensor.shape:
                    compatible_sd[key] = tensor
            new_model.load_state_dict(compatible_sd, strict=False)
            self.runtime.model = new_model
            self.runtime.mcfg = new_cfg
            self.runtime.tcfg.block_size = min(self.runtime.tcfg.block_size, new_cfg.block_size)
            self.runtime.optimizer = self.runtime._build_optimizer()
            self.runtime.scheduler = self.runtime._build_scheduler()
            return {"updated": "model_architecture", "loaded_params": len(compatible_sd)}

        if patch.patch_type == "tokenizer":
            merges = p.get("merges", [])
            if not isinstance(merges, list):
                raise ValueError("tokenizer merges must be a list")
            self.runtime.dataset.tokenizer.update_merges([(str(a), str(b)) for a, b in merges])
            self.runtime.dataset.rebuild_after_tokenizer_update()
            return {"updated": "tokenizer_merges", "merges_count": len(merges)}

        if patch.patch_type == "data":
            intensity = max(0.0, min(float(p.get("intensity", 0.0)), tcfg.max_augmented_fraction))
            pattern = str(p.get("pattern", "case_flip"))
            result = self.runtime.dataset.apply_augmentation(intensity=intensity, pattern=pattern)
            return {"updated": "dataset_augmentation", **result}

        if patch.patch_type == "loss":
            loss_name = str(p.get("loss_name", "cross_entropy"))
            if loss_name not in {"cross_entropy", "focal"}:
                raise ValueError("Unsupported loss_name")
            self.runtime.loss_name = loss_name
            if "label_smoothing" in p:
                tcfg.label_smoothing = max(0.0, min(0.2, float(p["label_smoothing"])))
            if "focal_gamma" in p:
                tcfg.focal_gamma = max(0.0, min(4.0, float(p["focal_gamma"])))
            return {"updated": "loss_fn", "loss_name": loss_name}

        if patch.patch_type == "sabotage":
            intensity = max(0.0, min(float(p.get("stress_noise_std", 0.0)), tcfg.max_sabotage_intensity))
            tcfg.stress_noise_std = intensity
            tcfg.sabotage_eval_only = bool(p.get("sabotage_eval_only", True))
            return {"updated": "stress_config", "stress_noise_std": intensity}

        if patch.patch_type == "resilience":
            if "stress_noise_std" in p:
                tcfg.stress_noise_std = max(0.0, min(float(p["stress_noise_std"]), 0.3))
            return {"updated": "resilience_config"}

        if patch.patch_type == "speed":
            if "compile_model" in p:
                tcfg.compile_model = bool(p["compile_model"])
            if "num_workers" in p:
                tcfg.num_workers = max(0, min(8, int(p["num_workers"])))
            if "pin_memory" in p:
                tcfg.pin_memory = bool(p["pin_memory"])
            if "scheduler_name" in p:
                tcfg.scheduler_name = str(p["scheduler_name"])
            self.runtime.optimizer = self.runtime._build_optimizer()
            self.runtime.scheduler = self.runtime._build_scheduler()
            self.runtime._maybe_compile()
            return {"updated": "speed_runtime_flags"}

        if patch.patch_type == "memory":
            if "use_gradient_checkpointing" in p:
                tcfg.use_gradient_checkpointing = bool(p["use_gradient_checkpointing"])
            if "amp_enabled" in p:
                tcfg.amp_enabled = bool(p["amp_enabled"])
            if "batch_size" in p:
                tcfg.batch_size = max(4, min(128, int(p["batch_size"])))
            if "micro_batch_size" in p:
                tcfg.micro_batch_size = max(4, min(tcfg.batch_size, int(p["micro_batch_size"])))
            self.runtime.scaler = torch.amp.GradScaler("cuda", enabled=self.runtime._use_fp16_scaler())
            return {"updated": "memory_runtime_flags"}

        if patch.patch_type in {"evaluation", "arbitration"}:
            return {"updated": "noop_control_patch"}

        raise ValueError(f"Unsupported patch type: {patch.patch_type}")


@dataclass
class AgentState:
    agent_id: str
    role: str
    alive: bool = True
    consecutive_failures: int = 0
    accepted_patches: int = 0
    rejected_patches: int = 0
    total_patches: int = 0
    total_votes_cast: int = 0
    total_runtime_sec: float = 0.0
    deaths: int = 0
    reassignments: int = 0
    last_error: str = ""
    capabilities: str = ""


class Agent(ABC):
    def __init__(self, agent_id: str, role: str, seed: int):
        self.agent_id = agent_id
        self.role = role
        self.seed = seed
        self.local_rng = random.Random(seed)
        self.state = AgentState(agent_id=agent_id, role=role, capabilities=self.describe_capabilities())

    @abstractmethod
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        raise NotImplementedError

    @abstractmethod
    def describe_capabilities(self) -> str:
        raise NotImplementedError

    def score_patch(self, patch: Patch, runtime_metrics: Dict[str, Any], runtime_state: Dict[str, Any]) -> PatchVote:
        approve = True
        score = 60.0
        reason = "Default neutral vote."
        val_loss = float(runtime_metrics.get("val_loss", 99.0))
        throughput = float(runtime_metrics.get("throughput_tokens_per_sec", 0.0))
        if patch.patch_type in {"hyperparam", "loss", "memory", "speed"}:
            score += 8.0
        if patch.patch_type == "sabotage":
            score -= 15.0
            approve = val_loss < 3.0
            reason = "Sabotage accepted only under controlled regime."
        if patch.patch_type == "model_arch":
            n_layer = int(runtime_state["model_config"]["n_layer"])
            if n_layer >= 12 and patch.payload.get("n_layer_delta", 0) > 0:
                score -= 10.0
                approve = False
                reason = "Architecture already near upper complexity bound."
        if throughput < 1000 and patch.patch_type in {"speed", "memory"}:
            score += 10.0
        score += self.local_rng.uniform(-5.0, 5.0)
        score = max(0.0, min(100.0, score))
        approve = approve and score >= 55.0
        return PatchVote(
            patch_id=patch.id,
            voter_id=self.agent_id,
            voter_role=self.role,
            approve=approve,
            score=score,
            reason=reason,
        )


class BugHunterAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        num_workers = int(runtime_state["training_config"]["num_workers"])
        payload = {
            "pin_memory": True,
            "num_workers": max(0, min(8, num_workers + self.local_rng.choice([-1, 0, 1]))),
            "persistent_workers": bool(num_workers > 0),
            "prefetch_factor": self.local_rng.choice([2, 4]),
        }
        return build_patch(self, cycle_index, "bugfix", payload, "Tune loader flags to reduce stalls and unstable worker churn.")

    def describe_capabilities(self) -> str:
        return "Detects training loop inefficiencies, unstable worker settings, and latent dead-config paths."


class HyperparamTunerAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        tcfg = runtime_state["training_config"]
        lr = float(tcfg["learning_rate"]) * self.local_rng.choice([0.85, 0.92, 1.05, 1.12])
        wd = float(tcfg["weight_decay"]) * self.local_rng.choice([0.8, 0.95, 1.05])
        batch_size = int(tcfg["batch_size"]) + self.local_rng.choice([-8, 0, 8])
        payload = {
            "learning_rate": max(5e-5, min(8e-4, lr)),
            "min_learning_rate": max(1e-5, min(2e-4, lr * 0.2)),
            "weight_decay": max(0.0, min(0.3, wd)),
            "dropout": max(0.0, min(0.35, float(tcfg["dropout"]) + self.local_rng.choice([-0.03, 0.0, 0.03]))),
            "batch_size": max(8, min(128, batch_size)),
            "micro_batch_size": max(8, min(128, batch_size)),
            "warmup_iters": max(20, min(500, int(tcfg["warmup_iters"]) + self.local_rng.choice([-25, 0, 25]))),
            "scheduler_name": self.local_rng.choice(["cosine", "linear", "none"]),
        }
        return build_patch(self, cycle_index, "hyperparam", payload, "Search bounded optimizer and schedule parameters.")

    def describe_capabilities(self) -> str:
        return "Optimizes LR, batch size, regularization, and scheduling under bounded safe ranges."


class LayerArchitectAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        payload = {
            "n_layer_delta": self.local_rng.choice([-1, 0, 1]),
            "n_head_delta": self.local_rng.choice([0, 1]),
            "n_embd_delta": self.local_rng.choice([-64, 0, 64]),
            "use_gru_gate": self.local_rng.choice([False, True]),
            "residual_scale": self.local_rng.choice([0.9, 1.0, 1.1]),
        }
        return build_patch(self, cycle_index, "model_arch", payload, "Adjust model depth/width and optional gated residual path.")

    def describe_capabilities(self) -> str:
        return "Refactors Transformer topology while preserving tensor compatibility constraints."


class TokenizerOptimizerAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        merges = [("t", "h"), ("i", "n"), ("e", "r")]
        self.local_rng.shuffle(merges)
        payload = {"merges": merges[: self.local_rng.choice([1, 2, 3])]}
        return build_patch(self, cycle_index, "tokenizer", payload, "Introduce bounded merge metadata for tokenizer experimentation.")

    def describe_capabilities(self) -> str:
        return "Explores tokenizer merge strategies without corrupting dataset alignment."


class DataAugmentorAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        payload = {
            "intensity": self.local_rng.choice([0.01, 0.03, 0.05, 0.08]),
            "pattern": self.local_rng.choice(["case_flip", "punctuation", "space_noise"]),
        }
        return build_patch(self, cycle_index, "data", payload, "Apply bounded text augmentation to improve robustness and generalization.")

    def describe_capabilities(self) -> str:
        return "Generates safe textual perturbations under strict augmentation budget."


class LossEngineerAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        use_focal = self.local_rng.choice([False, True])
        payload = {
            "loss_name": "focal" if use_focal else "cross_entropy",
            "label_smoothing": self.local_rng.choice([0.0, 0.02, 0.05, 0.08]),
            "focal_gamma": self.local_rng.choice([0.0, 1.0, 1.5, 2.0]),
        }
        return build_patch(self, cycle_index, "loss", payload, "Swap or blend loss shaping to stabilize confidence and rare-token learning.")

    def describe_capabilities(self) -> str:
        return "Designs and tunes loss functions, including focal weighting and label smoothing."


class SaboteurAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        payload = {
            "stress_noise_std": self.local_rng.choice([0.02, 0.05, 0.08, 0.12]),
            "sabotage_eval_only": True,
        }
        return build_patch(self, cycle_index, "sabotage", payload, "Inject controlled evaluation-only stress to verify resilience boundaries.")

    def describe_capabilities(self) -> str:
        return "Injects bounded faults to probe robustness without corrupting persistent checkpoints."


class ResilienceCheckerAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        current = float(runtime_state["training_config"]["stress_noise_std"])
        payload = {
            "stress_noise_std": max(0.0, min(0.15, current + self.local_rng.choice([-0.02, 0.0, 0.02]))),
        }
        return build_patch(self, cycle_index, "resilience", payload, "Tune stress-test profile to maintain robust validation envelope.")

    def describe_capabilities(self) -> str:
        return "Evaluates survivability under bounded perturbation and hardware/software noise models."

    def score_patch(self, patch: Patch, runtime_metrics: Dict[str, Any], runtime_state: Dict[str, Any]) -> PatchVote:
        score = 65.0
        reason = "Robustness-biased vote."
        if runtime_metrics.get("perplexity", 999.0) < 12.0:
            score += 10.0
        if patch.patch_type == "sabotage":
            score -= 10.0
        if patch.patch_type in {"memory", "speed", "bugfix"}:
            score += 6.0
        score += self.local_rng.uniform(-3.0, 3.0)
        approve = score >= 60.0
        return PatchVote(
            patch_id=patch.id,
            voter_id=self.agent_id,
            voter_role=self.role,
            approve=approve,
            score=max(0.0, min(100.0, score)),
            reason=reason,
        )


class SpeedDemonAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        payload = {
            "compile_model": self.local_rng.choice([False, True]),
            "num_workers": self.local_rng.choice([0, 2, 4, 6]),
            "pin_memory": True,
            "scheduler_name": self.local_rng.choice(["cosine", "linear"]),
        }
        return build_patch(self, cycle_index, "speed", payload, "Push runtime throughput with compile and loader tuning.")

    def describe_capabilities(self) -> str:
        return "Optimizes token throughput via compile, loader, and scheduling adjustments."

    def score_patch(self, patch: Patch, runtime_metrics: Dict[str, Any], runtime_state: Dict[str, Any]) -> PatchVote:
        score = 62.0 + (runtime_metrics.get("throughput_tokens_per_sec", 0.0) / 5000.0)
        if patch.patch_type == "speed":
            score += 10.0
        if patch.patch_type == "model_arch":
            score -= 6.0
        score += self.local_rng.uniform(-4.0, 4.0)
        approve = score >= 62.0
        return PatchVote(
            patch_id=patch.id,
            voter_id=self.agent_id,
            voter_role=self.role,
            approve=approve,
            score=max(0.0, min(100.0, score)),
            reason="Throughput-sensitive vote.",
        )


class MemoryWardenAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        batch_size = int(runtime_state["training_config"]["batch_size"])
        payload = {
            "use_gradient_checkpointing": self.local_rng.choice([False, True]),
            "amp_enabled": True,
            "batch_size": max(8, min(128, batch_size + self.local_rng.choice([-8, 0, 8]))),
            "micro_batch_size": max(8, min(64, batch_size)),
        }
        return build_patch(self, cycle_index, "memory", payload, "Reduce memory pressure while preserving training validity.")

    def describe_capabilities(self) -> str:
        return "Controls memory footprint using AMP, checkpointing, and bounded batch adaptation."

    def score_patch(self, patch: Patch, runtime_metrics: Dict[str, Any], runtime_state: Dict[str, Any]) -> PatchVote:
        mem = runtime_metrics.get("max_memory_allocated", 0)
        score = 60.0 + (10.0 if mem > 0 else 0.0)
        if patch.patch_type == "memory":
            score += 12.0
        score += self.local_rng.uniform(-4.0, 4.0)
        approve = score >= 60.0
        return PatchVote(
            patch_id=patch.id,
            voter_id=self.agent_id,
            voter_role=self.role,
            approve=approve,
            score=max(0.0, min(100.0, score)),
            reason="Memory-stability vote.",
        )


class EvaluatorAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        payload = {"priority_metric": self.local_rng.choice(["val_loss", "perplexity", "throughput", "bleu_like"])}
        return build_patch(self, cycle_index, "evaluation", payload, "Publish evaluation preference metadata for this cycle.")

    def describe_capabilities(self) -> str:
        return "Scores candidate patches using validation loss, perplexity, BLEU-like precision, and throughput."

    def score_patch(self, patch: Patch, runtime_metrics: Dict[str, Any], runtime_state: Dict[str, Any]) -> PatchVote:
        val_loss = float(runtime_metrics.get("val_loss", 99.0))
        perplexity = float(runtime_metrics.get("perplexity", 999.0))
        throughput = float(runtime_metrics.get("throughput_tokens_per_sec", 0.0))
        bleu_like = float(runtime_metrics.get("bleu_like", 0.0))
        base = 35.0
        if patch.patch_type in {"hyperparam", "loss", "memory", "speed"}:
            base += 12.0
        if patch.patch_type == "model_arch":
            base += 10.0
        if patch.patch_type in {"sabotage", "resilience"}:
            base -= 25.0
        val_loss_bonus = max(0.0, 45.0 - val_loss * 12.0)
        ppl_bonus = max(0.0, 8.0 - math.log(max(1.0, perplexity)) * 2.0)
        throughput_bonus = min(4.0, throughput / 20000.0)
        bleu_bonus = min(3.0, bleu_like / 50.0)
        score = base + val_loss_bonus + ppl_bonus + throughput_bonus + bleu_bonus + self.local_rng.uniform(-1.0, 1.0)
        approve = score >= 72.0 and patch.patch_type not in {"sabotage", "resilience"}
        return PatchVote(
            patch_id=patch.id,
            voter_id=self.agent_id,
            voter_role=self.role,
            approve=approve,
            score=max(0.0, min(100.0, score)),
            reason="Val-loss-dominant evaluator vote.",
        )


class ArbiterAgent(Agent):
    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        payload = {"max_patches": runtime_state["swarm_config"]["max_patches_applied_per_cycle"]}
        return build_patch(self, cycle_index, "arbitration", payload, "Declare arbitration capacity and patch conflict resolution policy.")

    def describe_capabilities(self) -> str:
        return "Resolves conflicts, computes weighted voting, applies patches, and manages lifecycle events."


def build_patch(agent: Agent, cycle_index: int, patch_type: str, payload: Dict[str, Any], summary: str) -> Patch:
    patch_id = stable_hash({
        "agent_id": agent.agent_id,
        "role": agent.role,
        "cycle_index": cycle_index,
        "patch_type": patch_type,
        "payload": payload,
    })[:24]
    return Patch(
        id=patch_id,
        author_id=agent.agent_id,
        role=agent.role,
        timestamp=utc_now(),
        patch_type=patch_type,
        payload=payload,
        summary=summary,
        cycle_index=cycle_index,
    )


ROLE_TO_CLASS: Dict[str, type[Agent]] = {
    "BugHunter": BugHunterAgent,
    "HyperparamTuner": HyperparamTunerAgent,
    "LayerArchitect": LayerArchitectAgent,
    "TokenizerOptimizer": TokenizerOptimizerAgent,
    "DataAugmentor": DataAugmentorAgent,
    "LossEngineer": LossEngineerAgent,
    "Saboteur": SaboteurAgent,
    "ResilienceChecker": ResilienceCheckerAgent,
    "SpeedDemon": SpeedDemonAgent,
    "MemoryWarden": MemoryWardenAgent,
    "Evaluator": EvaluatorAgent,
    "Arbiter": ArbiterAgent,
}

class LocalAgentWorker:
    def __init__(self, agent: Agent):
        self.agent = agent

    def generate_patch(self, runtime_state: Dict[str, Any], cycle_index: int) -> Patch:
        return self.agent.generate_patch(runtime_state, cycle_index)

    def score_patch(self, patch: Patch, runtime_metrics: Dict[str, Any], runtime_state: Dict[str, Any]) -> PatchVote:
        return self.agent.score_patch(patch, runtime_metrics, runtime_state)

    def get_state(self) -> AgentState:
        return self.agent.state


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


class SwarmEngine:
    def __init__(self, train_cfg: TrainingConfig, model_cfg: ModelConfig, swarm_cfg: SwarmConfig):
        swarm_cfg.validate()
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.swarm_cfg = swarm_cfg
        self.output_dir = ensure_dir(swarm_cfg.output_dir)
        self.logs_dir = ensure_dir(self.output_dir / "logs")
        self.metrics_path = self.logs_dir / "metrics.jsonl"
        self.decisions_path = self.logs_dir / "patch_decisions.jsonl"
        self.lifecycle_path = self.logs_dir / "lifecycle.jsonl"
        self.csv_path = self.logs_dir / "cycles.csv"
        self.logger = self._build_logger()
        set_global_seed(train_cfg.seed)
        self.dataset = TinyShakespeareData(swarm_cfg.data_dir, train_cfg, self.logger)
        self.model_cfg.vocab_size = self.dataset.tokenizer.vocab_size
        self.model_cfg.block_size = train_cfg.block_size
        self.runtime = TrainingRuntime(train_cfg, model_cfg, self.dataset, self.output_dir, self.logger)
        self.patch_applier = PatchApplier(self.runtime, self.logger, swarm_cfg)
        self.parallel = ParallelBackend(use_ray=swarm_cfg.use_ray, logger=self.logger)
        self.workers: Dict[str, Any] = {}
        self.agent_states: Dict[str, AgentState] = {}
        self.role_to_agent_ids: Dict[str, List[str]] = defaultdict(list)
        self.stop_requested = False
        self._install_signal_handlers()
        self._init_agents()
        self._init_csv()

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("swarm_forge")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        return logger

    def _rebuild_file_handler(self) -> None:
        self.logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        fh = logging.FileHandler(self.logs_dir / "swarm.log", encoding="utf-8")
        fh.setFormatter(fmt)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def _install_signal_handlers(self) -> None:
        def _handler(signum: int, frame: Any) -> None:
            self.stop_requested = True
            self.logger.warning("Signal received: %s. Shutdown requested.", signum)

        signal.signal(signal.SIGINT, _handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handler)

    def _init_agents(self) -> None:
        self._rebuild_file_handler()
        idx = 0
        for role in ROLE_NAMES:
            for n in range(self.swarm_cfg.agents_per_role):
                agent_id = f"{role}-{n:02d}"
                worker = self.parallel.create_worker(role_name=role, agent_id=agent_id, seed=self.train_cfg.seed + idx)
                idx += 1
                st = self.parallel.get_state(worker)
                self.workers[agent_id] = worker
                self.agent_states[agent_id] = st
                self.role_to_agent_ids[role].append(agent_id)
        if len(self.workers) != 120:
            raise RuntimeError(f"Expected 120 agents, got {len(self.workers)}.")
        self.logger.info("Initialized %d agents across %d roles.", len(self.workers), len(self.role_to_agent_ids))

    def _init_csv(self) -> None:
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=[
                    "cycle_index", "timestamp", "train_loss", "val_loss", "perplexity", "bleu_like",
                    "throughput_tokens_per_sec", "agents_alive", "accepted_patches", "rejected_patches",
                    "global_step", "best_val_loss"
                ])
                writer.writeheader()

    def runtime_state_dict(self) -> Dict[str, Any]:
        return {
            "training_config": asdict(self.runtime.tcfg),
            "model_config": asdict(self.runtime.mcfg),
            "last_eval_metrics": dict(self.runtime.last_eval_metrics),
            "swarm_config": asdict(self.swarm_cfg),
            "dataset": {
                "vocab_size": self.dataset.tokenizer.vocab_size,
                "merges": list(self.dataset.tokenizer.merges),
                "text_length": len(self.dataset.text),
            },
        }

    def _update_agent_state(self, agent_id: str, updater: Callable[[AgentState], None]) -> None:
        st = self.agent_states[agent_id]
        updater(st)
        self.parallel.set_state(self.workers[agent_id], st)

    def _mark_agent_failure(self, agent_id: str, error: str) -> None:
        def _apply(st: AgentState) -> None:
            st.consecutive_failures += 1
            st.last_error = error
        self._update_agent_state(agent_id, _apply)
        st = self.agent_states[agent_id]
        if st.consecutive_failures >= self.swarm_cfg.dead_after_failures and st.alive:
            self._kill_agent(agent_id, reason=error)

    def _mark_agent_success(self, agent_id: str, runtime_sec: float) -> None:
        def _apply(st: AgentState) -> None:
            st.consecutive_failures = 0
            st.total_runtime_sec += runtime_sec
            st.total_patches += 1
            st.last_error = ""
        self._update_agent_state(agent_id, _apply)

    def _kill_agent(self, agent_id: str, reason: str) -> None:
        st = self.agent_states[agent_id]
        if not st.alive:
            return
        st.alive = False
        st.deaths += 1
        append_jsonl(self.lifecycle_path, {
            "event": "death",
            "timestamp": utc_now(),
            "agent_id": agent_id,
            "role": st.role,
            "reason": reason,
        })
        self.logger.warning("Agent died: %s role=%s reason=%s", agent_id, st.role, reason)
        self._reassign_role_lottery(agent_id, st.role)

    def _reassign_role_lottery(self, dead_agent_id: str, role: str) -> None:
        living = [aid for aid in self.role_to_agent_ids[role] if aid != dead_agent_id and self.agent_states[aid].alive]
        if not living:
            append_jsonl(self.lifecycle_path, {
                "event": "reassign_skipped",
                "timestamp": utc_now(),
                "dead_agent_id": dead_agent_id,
                "role": role,
                "reason": "no_living_agent_in_role",
            })
            self.logger.error("No living agents available for reassignment in role=%s", role)
            return
        chosen = random.choice(living)
        chosen_state = self.agent_states[chosen]
        chosen_state.reassignments += 1
        append_jsonl(self.lifecycle_path, {
            "event": "role_reassigned",
            "timestamp": utc_now(),
            "dead_agent_id": dead_agent_id,
            "role": role,
            "assignee_agent_id": chosen,
        })
        self.parallel.set_state(self.workers[chosen], chosen_state)
        self.logger.info("Role lottery reassignment: dead=%s role=%s -> assignee=%s", dead_agent_id, role, chosen)

    def _collect_patches(self, cycle_index: int) -> List[Patch]:
        patches: List[Patch] = []
        runtime_state = self.runtime_state_dict()
        for agent_id, worker in self.workers.items():
            st = self.agent_states[agent_id]
            if not st.alive:
                continue
            start = time.perf_counter()
            try:
                patch = self.parallel.generate_patch(worker, runtime_state, cycle_index)
                elapsed = time.perf_counter() - start
                self._mark_agent_success(agent_id, elapsed)
                patches.append(patch)
            except Exception as exc:
                self.logger.error("Patch generation failed: agent=%s error=%s", agent_id, exc)
                self._mark_agent_failure(agent_id, str(exc))
        if len(patches) != sum(1 for st in self.agent_states.values() if st.alive):
            self.logger.warning("Patch collection mismatch: patches=%d alive_agents=%d", len(patches), sum(1 for st in self.agent_states.values() if st.alive))
        return patches

    def _agent_vote_weight(self, role: str) -> float:
        if role == "Evaluator":
            return self.swarm_cfg.evaluator_weight_total / 10.0
        if getattr(self.swarm_cfg, "reduced_roles_mode", False):
            low_signal_roles = {"Saboteur", "ResilienceChecker", "TokenizerOptimizer", "DataAugmentor"}
            if role in low_signal_roles:
                return 0.10
        return self.swarm_cfg.others_weight_total / 110.0

    def _collect_votes(self, patches: List[Patch], runtime_metrics: Dict[str, Any]) -> Dict[str, List[PatchVote]]:
        votes_by_patch: Dict[str, List[PatchVote]] = defaultdict(list)
        runtime_state = self.runtime_state_dict()
        for patch in patches:
            for agent_id, worker in self.workers.items():
                st = self.agent_states[agent_id]
                if not st.alive:
                    continue
                start = time.perf_counter()
                try:
                    vote = self.parallel.score_patch(worker, patch, runtime_metrics, runtime_state)
                    vote.weight = self._agent_vote_weight(vote.voter_role)
                    votes_by_patch[patch.id].append(vote)
                    st.total_votes_cast += 1
                    st.total_runtime_sec += time.perf_counter() - start
                    self.parallel.set_state(worker, st)
                except Exception as exc:
                    self.logger.error("Vote generation failed: voter=%s patch=%s error=%s", agent_id, patch.id, exc)
                    self._mark_agent_failure(agent_id, str(exc))
        return votes_by_patch

    def _decide_patches(self, patches: List[Patch], votes_by_patch: Dict[str, List[PatchVote]]) -> List[PatchDecision]:
        decisions: List[PatchDecision] = []
        best_by_group: Dict[str, Tuple[PatchDecision, float]] = {}
        for patch in patches:
            votes = votes_by_patch.get(patch.id, [])
            if not votes:
                decision = PatchDecision(
                    patch_id=patch.id,
                    approval_percent=0.0,
                    weighted_score=0.0,
                    applied=False,
                    reason="no_votes",
                    conflict_group=PatchConflictResolver.conflict_group(patch),
                    summary=patch.summary,
                    author_id=patch.author_id,
                    role=patch.role,
                    patch_type=patch.patch_type,
                )
                decisions.append(decision)
                continue
            total_weight = sum(v.weight for v in votes)
            if total_weight <= 0:
                total_weight = 1.0
            approval_weight = sum(v.weight for v in votes if v.approve)
            approval_percent = 100.0 * approval_weight / total_weight
            weighted_score = sum(v.score * v.weight for v in votes) / total_weight
            apply_candidate = (
                approval_percent >= self.swarm_cfg.patch_apply_approval_threshold
                and weighted_score >= self.swarm_cfg.patch_apply_score_threshold
            )
            group = PatchConflictResolver.conflict_group(patch)
            reason = "thresholds_met" if apply_candidate else "thresholds_not_met"
            decision = PatchDecision(
                patch_id=patch.id,
                approval_percent=approval_percent,
                weighted_score=weighted_score,
                applied=False,
                reason=reason,
                conflict_group=group,
                summary=patch.summary,
                author_id=patch.author_id,
                role=patch.role,
                patch_type=patch.patch_type,
            )
            decisions.append(decision)
            if apply_candidate:
                existing = best_by_group.get(group)
                rank = (weighted_score, approval_percent)
                if existing is None or rank > existing[1]:
                    best_by_group[group] = (decision, weighted_score * 1000.0 + approval_percent)

        accepted_ids = [d.patch_id for d, _ in best_by_group.values()]
        accepted_ids = sorted(
            accepted_ids,
            key=lambda pid: next(d.weighted_score for d in decisions if d.patch_id == pid),
            reverse=True,
        )[: self.swarm_cfg.max_patches_applied_per_cycle]

        for d in decisions:
            if d.patch_id in accepted_ids:
                d.applied = True
                d.reason = "accepted_best_in_conflict_group"
            elif d.reason == "thresholds_met":
                d.reason = "rejected_due_to_conflict_or_limit"

        decisions_path = getattr(self, "decisions_path", None)
        if decisions_path is not None:
            for d in decisions:
                append_jsonl(decisions_path, {
                    "timestamp": utc_now(),
                    **asdict(d),
                })
        return decisions

    def _apply_accepted_patches(self, patches: List[Patch], decisions: List[PatchDecision]) -> List[AppliedPatchRecord]:
        patch_map = {p.id: p for p in patches}
        accepted = [d for d in decisions if d.applied]
        priority = {
            "bugfix": 1,
            "hyperparam": 2,
            "loss": 3,
            "memory": 4,
            "speed": 5,
            "data": 6,
            "tokenizer": 7,
            "resilience": 8,
            "sabotage": 9,
            "model_arch": 10,
            "evaluation": 11,
            "arbitration": 12,
        }
        accepted.sort(key=lambda d: priority.get(d.patch_type, 999))
        applied_records: List[AppliedPatchRecord] = []
        for d in accepted:
            patch = patch_map[d.patch_id]
            record = self.patch_applier.apply_patch(patch)
            applied_records.append(record)
            author_state = self.agent_states[patch.author_id]
            if record.success:
                author_state.accepted_patches += 1
            else:
                author_state.rejected_patches += 1
            self.parallel.set_state(self.workers[patch.author_id], author_state)
        return applied_records

    def _post_train_checkpoint(self, cycle_index: int, metrics: Dict[str, Any]) -> Optional[Path]:
        val_loss = float(metrics.get("val_loss", float("inf")))
        if val_loss < self.runtime.best_val_loss:
            self.runtime.best_val_loss = val_loss
            tag = f"best_cycle_{cycle_index:04d}_valloss_{val_loss:.4f}"
            path = self.runtime.save_checkpoint(tag, metrics)
            self.logger.info("Saved improved checkpoint: %s", path)
            return path
        if cycle_index % 4 == 0:
            tag = f"cycle_{cycle_index:04d}"
            return self.runtime.save_checkpoint(tag, metrics)
        return None

    def _write_cycle_csv(self, cycle_index: int, metrics: Dict[str, Any], accepted_count: int, rejected_count: int) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "cycle_index", "timestamp", "train_loss", "val_loss", "perplexity", "bleu_like",
                "throughput_tokens_per_sec", "agents_alive", "accepted_patches", "rejected_patches",
                "global_step", "best_val_loss"
            ])
            writer.writerow({
                "cycle_index": cycle_index,
                "timestamp": utc_now(),
                "train_loss": metrics.get("train_loss"),
                "val_loss": metrics.get("val_loss"),
                "perplexity": metrics.get("perplexity"),
                "bleu_like": metrics.get("bleu_like"),
                "throughput_tokens_per_sec": metrics.get("throughput_tokens_per_sec"),
                "agents_alive": sum(1 for st in self.agent_states.values() if st.alive),
                "accepted_patches": accepted_count,
                "rejected_patches": rejected_count,
                "global_step": self.runtime.global_step,
                "best_val_loss": self.runtime.best_val_loss,
            })

    def run_cycle(self, cycle_index: int) -> Dict[str, Any]:
        cycle_start = time.perf_counter()
        self.logger.info("Cycle %04d started.", cycle_index)

        baseline_metrics = self.runtime.evaluate()
        baseline_val_loss = float(baseline_metrics.get("val_loss", float("inf")))
        patches = self._collect_patches(cycle_index=cycle_index)
        if self.swarm_cfg.max_patch_eval_candidates is not None:
            patches = patches[: self.swarm_cfg.max_patch_eval_candidates]
        votes_by_patch = self._collect_votes(patches, runtime_metrics=baseline_metrics)
        decisions = self._decide_patches(patches, votes_by_patch)
        applied_records = self._apply_accepted_patches(patches, decisions)

        train_stats = self.runtime.train_steps(self.runtime.tcfg.max_iters_per_cycle)
        metrics = self.runtime.evaluate()
        metrics["train_loss"] = float(train_stats["train_loss_recent"])
        metrics["train_throughput_tokens_per_sec"] = float(train_stats["train_throughput_tokens_per_sec"])
        metrics["global_step"] = int(train_stats["global_step"])
        metrics["applied_patch_count"] = sum(1 for r in applied_records if r.success)
        metrics["rejected_patch_count"] = sum(1 for d in decisions if not d.applied)
        metrics["alive_by_role"] = count_alive_by_role(list(self.agent_states.values()))
        metrics["baseline_val_loss"] = baseline_val_loss
        metrics["val_loss_delta"] = float(metrics["val_loss"] - baseline_val_loss)
        metrics["val_loss_improved"] = bool(metrics["val_loss"] < baseline_val_loss)

        ckpt_path = self._post_train_checkpoint(cycle_index, metrics)
        elapsed = time.perf_counter() - cycle_start
        if elapsed < self.swarm_cfg.cycle_seconds:
            remaining = self.swarm_cfg.cycle_seconds - elapsed
            self.logger.info("Cycle %04d finished early; sleeping %.2f sec to honor logical window.", cycle_index, remaining)
            slept = 0.0
            while slept < remaining and not self.stop_requested:
                nap = min(self.swarm_cfg.quiet_sleep, remaining - slept)
                time.sleep(nap)
                slept += nap
        else:
            self.logger.warning("Cycle %04d exceeded logical window by %.2f sec.", cycle_index, elapsed - self.swarm_cfg.cycle_seconds)

        cycle_payload = {
            "timestamp": utc_now(),
            "cycle_index": cycle_index,
            "baseline_metrics": baseline_metrics,
            "metrics": metrics,
            "patches": [asdict(p) for p in patches],
            "decisions": [asdict(d) for d in decisions],
            "applied_records": [asdict(r) for r in applied_records],
            "checkpoint_path": str(ckpt_path) if ckpt_path is not None else None,
        }
        append_jsonl(self.metrics_path, cycle_payload)
        self._write_cycle_csv(
            cycle_index=cycle_index,
            metrics=metrics,
            accepted_count=sum(1 for r in applied_records if r.success),
            rejected_count=sum(1 for d in decisions if not d.applied),
        )

        winner_summaries = [d for d in decisions if d.applied]
        self.logger.info(
            "Cycle %04d summary | train_loss=%s val_loss=%s delta=%s ppl=%s bleu=%s tok/s=%s alive=%d winners=%d",
            cycle_index,
            human_float(metrics["train_loss"]),
            human_float(metrics["val_loss"]),
            human_float(metrics["val_loss_delta"]),
            human_float(metrics["perplexity"]),
            human_float(metrics["bleu_like"]),
            human_float(metrics["throughput_tokens_per_sec"]),
            sum(1 for st in self.agent_states.values() if st.alive),
            len(winner_summaries),
        )
        for d in sorted(decisions, key=lambda x: (x.applied, x.weighted_score), reverse=True)[:12]:
            self.logger.info(
                "Patch %s | role=%s type=%s approve=%.2f%% score=%.2f applied=%s summary=%s",
                d.patch_id,
                d.role,
                d.patch_type,
                d.approval_percent,
                d.weighted_score,
                d.applied,
                d.summary,
            )
        return cycle_payload

    def run(self) -> None:
        start_wall = time.perf_counter()
        cycle_index = 1
        if self.swarm_cfg.max_cycles is not None:
            max_cycles = self.swarm_cfg.max_cycles
        else:
            max_cycles = max(1, int((self.swarm_cfg.max_hours * 3600) // self.swarm_cfg.cycle_seconds))
        self.logger.info("Swarm Forge started | device=%s cycles=%d output=%s", self.train_cfg.device, max_cycles, self.output_dir)
        try:
            while cycle_index <= max_cycles and not self.stop_requested:
                self.run_cycle(cycle_index)
                cycle_index += 1
        except KeyboardInterrupt:
            self.logger.warning("KeyboardInterrupt received; stopping.")
        finally:
            final_metrics = self.runtime.evaluate()
            final_ckpt = self.runtime.save_checkpoint("final", final_metrics)
            append_jsonl(self.metrics_path, {
                "timestamp": utc_now(),
                "event": "shutdown",
                "final_metrics": final_metrics,
                "final_checkpoint": str(final_ckpt),
                "runtime_seconds": time.perf_counter() - start_wall,
            })
            self.parallel.shutdown()
            self.logger.info("Swarm Forge shutdown complete. Final checkpoint: %s", final_ckpt)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Swarm Forge v1.0")
    parser.add_argument("--output-dir", type=str, default="runs/swarm_forge_v1")
    parser.add_argument("--data-dir", type=str, default="data/tinyshakespeare")
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--cycle-seconds", type=int, default=900)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-iters-per-cycle", type=int, default=800)
    parser.add_argument("--eval-iters", type=int, default=40)
    parser.add_argument("--use-ray", action="store_true", default=False)
    parser.add_argument("--compile-model", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--approval-threshold", type=float, default=60.0)
    parser.add_argument("--score-threshold", type=float, default=70.0)
    parser.add_argument("--focused-search", action="store_true", default=False)
    parser.add_argument("--patch-trial-train-steps", type=int, default=40)
    parser.add_argument("--reduced-roles-mode", action="store_true", default=False)
    parser.add_argument("--baseline-only", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_cfg = TrainingConfig(
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        compile_model=args.compile_model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        micro_batch_size=args.batch_size,
        block_size=args.block_size,
        max_iters_per_cycle=args.max_iters_per_cycle,
        eval_iters=args.eval_iters,
        dropout=args.dropout,
        amp_enabled=args.amp,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        focused_search=args.focused_search,
        patch_trial_train_steps=args.patch_trial_train_steps,
    )
    model_cfg = ModelConfig(
        vocab_size=256,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    swarm_cfg = SwarmConfig(
        cycle_seconds=args.cycle_seconds,
        max_hours=args.max_hours,
        max_cycles=args.max_cycles,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        use_ray=args.use_ray,
        patch_apply_approval_threshold=args.approval_threshold,
        patch_apply_score_threshold=args.score_threshold,
        reduced_roles_mode=args.reduced_roles_mode,
    )
    engine = SwarmEngine(train_cfg=train_cfg, model_cfg=model_cfg, swarm_cfg=swarm_cfg)
    if args.baseline_only:
        metrics = engine.runtime.evaluate()
        print(json.dumps({"mode": "baseline_only", "metrics": metrics}, sort_keys=True))
    else:
        engine.run()


if __name__ == "__main__":
    main()

# ============================================================
# INSTALLATION AND EXECUTION
# ============================================================
#
# Recommended environment:
#   Python 3.11+
#
# Recommended dependencies:
#   pip install "torch>=2.3.0" "ray[default]>=2.20.0"
#
# CPU-only fallback:
#   pip install "torch>=2.3.0" "ray[default]>=2.20.0"
#   Then run with: --device cpu
#
# Minimal run:
#   python swarm_forge.py --use-ray --amp --pin-memory --compile-model
#
# Explicit 24-hour run with default 15-minute cycles:
#   python swarm_forge.py --use-ray --max-hours 24 --cycle-seconds 900 --output-dir runs/swarm_forge_v1
#
# Short smoke test:
#   python swarm_forge.py --max-cycles 1 --cycle-seconds 30 --device cpu --batch-size 8 --n-layer 4 --n-head 4 --n-embd 128 --max-iters-per-cycle 10 --eval-iters 4
#
# TinyShakespeare dataset:
#   The script downloads TinyShakespeare automatically into:
#     data/tinyshakespeare/input.txt
#   If you want to pre-stage it manually:
#     mkdir -p data/tinyshakespeare
#     curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/tinyshakespeare/input.txt
#
# Useful runtime flags:
#   --device cuda
#   --dtype bfloat16
#   --batch-size 32
#   --block-size 256
#   --learning-rate 3e-4
#   --n-layer 6
#   --n-head 6
#   --n-embd 384
#   --num-workers 2
#   --approval-threshold 60
#   --score-threshold 70
#
# Output artifacts:
#   runs/swarm_forge_v1/
#     logs/swarm.log
#     logs/metrics.jsonl
#     logs/patch_decisions.jsonl
#     logs/lifecycle.jsonl
#     logs/cycles.csv
#     checkpoints/checkpoint_*.pt
#
# Notes:
#   - Ray is used when available and requested with --use-ray.
#   - If Ray initialization fails, the system automatically degrades to local worker execution.
#   - The system enforces exactly 120 agents, 12 roles, 10 agents per role.
#   - Each cycle generates exactly one patch per living agent and uses the required weighted voting rules.
#

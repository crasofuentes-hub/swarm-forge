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

def trust_region(param_name: str, proposed: float, current: float) -> float:
    TRUST_REGION = {
        "learning_rate": 0.20,
        "dropout": 0.15,
        "batch_size": 0.25,
    }

    delta = TRUST_REGION.get(param_name, 0.20)
    lower = current * (1 - delta)
    upper = current * (1 + delta)
    return max(lower, min(proposed, upper))


def golden_checkpoint_guard(val_loss: float, best_val_loss: Optional[float]) -> bool:
    if best_val_loss is None:
        return False
    return val_loss > best_val_loss * 1.15

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
WIKITEXT2_TRAIN_URL = "https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.train.raw"
WIKITEXT2_VALID_URL = "https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.valid.raw"
WIKITEXT2_TEST_URL = "https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.test.raw"


from .config import ModelConfig, SwarmConfig, TrainingConfig
from .common import append_jsonl, ensure_dir, stable_hash, utc_now
from .contracts import ExecutionResult, RuntimeSnapshot
from .patch_applier import PatchApplier
from .agents import AgentState, Agent, BugHunterAgent, HyperparamTunerAgent, LayerArchitectAgent, TokenizerOptimizerAgent, DataAugmentorAgent, LossEngineerAgent, SaboteurAgent, ResilienceCheckerAgent, SpeedDemonAgent, MemoryWardenAgent, EvaluatorAgent, ArbiterAgent, build_patch, ROLE_TO_CLASS, LocalAgentWorker

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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


from .data import CharTokenizer, TinyShakespeareData, WikiText2Data, build_dataset

from .proposals import AppliedPatchRecord, ExperimentProposal, build_experiment_proposal
from .patches import Patch, PatchVote, PatchDecision, PatchConflictResolver

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

    def snapshot_state(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            training_config=asdict(self.tcfg),
            model_config=asdict(self.mcfg),
            loss_name=self.loss_name,
            best_val_loss=self.best_val_loss,
            global_step=self.global_step,
            tokenizer_merges=list(self.dataset.tokenizer.merges),
            dataset_text_hash=stable_hash(self.dataset.text[:50000]),
            model_state_dict={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler is not None else None,
        )

    def restore_state(self, snapshot: RuntimeSnapshot) -> None:
        self.tcfg = TrainingConfig(**snapshot.training_config)
        self.mcfg = ModelConfig(**snapshot.model_config)
        self.loss_name = snapshot.loss_name
        self.best_val_loss = snapshot.best_val_loss
        self.global_step = snapshot.global_step
        self.dataset.tokenizer.update_merges(snapshot.tokenizer_merges)
        self.model = NanoGPT(self.mcfg).to(self.device)
        self.model.load_state_dict(snapshot.model_state_dict)
        self.model.gradient_checkpointing = self.tcfg.use_gradient_checkpointing
        self.optimizer = self._build_optimizer()
        self.optimizer.load_state_dict(snapshot.optimizer_state_dict)
        self.scheduler = self._build_scheduler()
        if self.scheduler is not None and snapshot.scheduler_state_dict is not None:
            self.scheduler.load_state_dict(snapshot.scheduler_state_dict)

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

    def load_checkpoint(self, path: str, reset_optimizer_scheduler: bool = False) -> Dict[str, Any]:
        """
        Load a previously saved checkpoint and restore model/optimizer/scheduler state.
        Designed for continuation on the same architecture family.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu")

        # Restore model config and rebuild model on current runtime device.
        self.mcfg = ModelConfig(**ckpt["model_config"])
        self.mcfg.block_size = int(self.tcfg.block_size)

        self.model = NanoGPT(self.mcfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.gradient_checkpointing = self.tcfg.use_gradient_checkpointing

        # Restore counters and loss name.
        self.loss_name = str(ckpt.get("loss_name", "cross_entropy"))
        self.best_val_loss = float(ckpt.get("best_val_loss", self.best_val_loss))
        self.global_step = int(ckpt.get("global_step", self.global_step))

        # Restore tokenizer merges if present.
        merges = ckpt.get("tokenizer_merges", None)
        if merges is not None:
            self.dataset.tokenizer.update_merges([(str(a), str(b)) for a, b in merges])
            self.dataset.rebuild_after_tokenizer_update()

        # Optimizer/scheduler state: best-effort load; if incompatible, continue with fresh state.
        self.optimizer = self._build_optimizer()
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            pass

        for pg in self.optimizer.param_groups:
            pg["lr"] = float(self.tcfg.learning_rate)

        self.logger.info("Effective optimizer lr after resume: %.8f", float(self.optimizer.param_groups[0]["lr"]))

        self.scheduler = self._build_scheduler()
        if self.scheduler is not None:
            try:
                ssd = ckpt.get("scheduler_state_dict", None)
                if ssd is not None:
                    self.scheduler.load_state_dict(ssd)
            except Exception:
                pass

        self.last_eval_metrics = dict(ckpt.get("last_eval_metrics", {}))
        return {
            "checkpoint": str(ckpt_path),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "loss_name": self.loss_name,
        }

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

    def train_steps(self, num_steps: int) -> ExecutionResult:
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
        return ExecutionResult(
            train_loss_recent=total_loss / max(1, num_steps),
            train_throughput_tokens_per_sec=float(throughput),
            global_step=self.global_step,
        )

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
        self.proposals_path = self.logs_dir / "experiment_proposals.jsonl"
        self.csv_path = self.logs_dir / "cycles.csv"
        self.logger = self._build_logger()
        set_global_seed(train_cfg.seed)
        self.dataset = build_dataset(swarm_cfg.dataset_name, swarm_cfg.data_dir, train_cfg, self.logger)
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


    def _append_experiment_proposal(self, proposal: ExperimentProposal) -> None:
        append_jsonl(self.proposals_path, {
            "timestamp": utc_now(),
            **asdict(proposal),
        })

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
                    best_by_group[group] = (decision, rank)

        accepted_ids = [d.patch_id for d, _ in best_by_group.values()]
        accepted_ids = sorted(
            accepted_ids,
            key=lambda pid: next(d.weighted_score for d in decisions if d.patch_id == pid),
            reverse=True,
        )[: self.swarm_cfg.max_patches_applied_per_cycle]

        patch_type_by_id = {p.id: p.patch_type for p in patches}

        for d in decisions:
            patch_type = patch_type_by_id.get(d.patch_id)

            if (
                d.patch_id in accepted_ids
                and not (
                    patch_type == "model_arch"
                    and self.runtime.best_val_loss <= 1.70
                )
            ):
                d.applied = True
                d.reason = "accepted_best_in_conflict_group"
            elif d.patch_id in accepted_ids and patch_type == "model_arch" and self.runtime.best_val_loss <= 1.70:
                d.applied = False
                d.reason = "architecture_frozen_in_finetune"
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

    def _guarded_train_cycle(self, total_steps: int, chunk_steps: int = 20, tol_abs: float = 0.002, tol_drift: float = 0.010) -> Dict[str, Any]:

        baseline_metrics = self.runtime.evaluate()
        baseline_val = float(baseline_metrics.get("val_loss", float("inf")))

        best_snapshot = self.runtime.snapshot_state()
        best_metrics = dict(baseline_metrics)
        best_val = baseline_val
        best_train_loss_recent = None
        steps_done = 0
        stopped_early = False
        stop_reason = "completed_with_best_snapshot_restored"

        while steps_done < total_steps:
            current_chunk = min(chunk_steps, total_steps - steps_done)

            train_stats = self.runtime.train_steps(current_chunk)
            metrics = self.runtime.evaluate()
            val = float(metrics.get("val_loss", float("inf")))
            steps_done += current_chunk

            self.logger.info(
                "Guarded chunk | steps_done=%d/%d train_loss_recent=%.4f val_loss=%.4f baseline_val=%.4f best_val=%.4f",
                steps_done,
                total_steps,
                float(train_stats.get("train_loss_recent", float("nan"))),
                val,
                baseline_val,
                best_val,
            )

            if val < best_val:
                best_val = val
                best_metrics = dict(metrics)
                best_snapshot = self.runtime.snapshot_state()
                best_train_loss_recent = float(train_stats.get("train_loss_recent", float("nan")))

            too_far_from_best = val > (best_val + tol_abs)
            too_far_from_baseline = val > (baseline_val + tol_drift)

            if too_far_from_best or too_far_from_baseline:
                stopped_early = True
                stop_reason = "rollback_guard_triggered"
                self.logger.info(
                    "Guarded stop | steps_done=%d val_loss=%.4f best_val=%.4f baseline_val=%.4f tol_abs=%.4f tol_drift=%.4f",
                    steps_done,
                    val,
                    best_val,
                    baseline_val,
                    tol_abs,
                    tol_drift,
                )
                break

        self.runtime.restore_state(best_snapshot)

        final_metrics = dict(best_metrics)
        if best_train_loss_recent is not None:
            final_metrics["train_loss"] = float(best_train_loss_recent)
        final_metrics["global_step"] = int(self.runtime.global_step)
        final_metrics["guarded_steps_done"] = int(steps_done)
        final_metrics["guarded_stopped_early"] = bool(stopped_early)
        final_metrics["guarded_stop_reason"] = stop_reason
        final_metrics["baseline_val_loss"] = float(baseline_val)
        final_metrics["val_loss_delta"] = float(final_metrics["val_loss"] - baseline_val)
        final_metrics["val_loss_improved"] = bool(final_metrics["val_loss"] < baseline_val)

        return final_metrics

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

        metrics = self._guarded_train_cycle(self.runtime.tcfg.max_iters_per_cycle)
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
    parser.add_argument("--dataset-name", type=str, default="tinyshakespeare")
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
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-reset-state", action="store_true", default=False)
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
        dataset_name=args.dataset_name,
        use_ray=args.use_ray,
        patch_apply_approval_threshold=args.approval_threshold,
        patch_apply_score_threshold=args.score_threshold,
        reduced_roles_mode=args.reduced_roles_mode,
    )
    engine = SwarmEngine(train_cfg=train_cfg, model_cfg=model_cfg, swarm_cfg=swarm_cfg)

    if args.resume:
        info = engine.runtime.load_checkpoint(args.resume, reset_optimizer_scheduler=bool(args.resume_reset_state))
        engine.logger.info(
            "Resumed from checkpoint=%s best_val_loss=%.4f global_step=%d loss=%s",
            info["checkpoint"],
            float(info["best_val_loss"]),
            int(info["global_step"]),
            str(info["loss_name"]),
        )
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

"""Patch application layer for Swarm Forge."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict

from .config import ModelConfig, SwarmConfig
from .patches import Patch
from .proposals import AppliedPatchRecord


class PatchApplier:
    def __init__(self, runtime: Any, logger: logging.Logger, swarm_cfg: SwarmConfig):
        self.runtime = runtime
        self.logger = logger
        self.swarm_cfg = swarm_cfg

    def validate_patch(self, patch: Patch) -> None:
        from . import core as _core

        if patch.patch_type not in _core.PATCH_TYPES:
            raise ValueError(f"Unknown patch type: {patch.patch_type}")
        if not isinstance(patch.payload, dict):
            raise ValueError("Patch payload must be a dict.")
        if not patch.summary.strip():
            raise ValueError("Patch summary must not be empty.")

    def apply_patch(self, patch: Patch) -> AppliedPatchRecord:
        from . import core as _core

        self.validate_patch(patch)
        snapshot = self.runtime.snapshot_state()
        try:
            details = self._apply_impl(patch)
            record = AppliedPatchRecord(
                patch_id=patch.id,
                patch_type=patch.patch_type,
                author_id=patch.author_id,
                role=patch.role,
                applied_at=_core.utc_now(),
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
                applied_at=_core.utc_now(),
                payload_hash=patch.payload_hash(),
                success=False,
                details={"error": str(exc)},
            )

    def _apply_impl(self, patch: Patch) -> Dict[str, Any]:
        from . import core as _core

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
            new_model = _core.NanoGPT(new_cfg).to(self.runtime.device)
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
            intensity = max(0.0, min(float(p.get("stress_noise_std", 0.0)), tcfg.max_sabotage_intensity))
            tcfg.stress_noise_std = intensity
            return {"updated": "resilience_config", "stress_noise_std": intensity}

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
            self.runtime.scaler = _core.torch.amp.GradScaler("cuda", enabled=self.runtime._use_fp16_scaler())
            return {"updated": "memory_runtime_flags"}

        if patch.patch_type in {"evaluation", "arbitration"}:
            return {"updated": "noop_control_patch"}

        raise ValueError(f"Unsupported patch type: {patch.patch_type}")


__all__ = [
    "PatchApplier",
]
"""Dataset and tokenization layer for Swarm Forge."""

from __future__ import annotations

from pathlib import Path

import json
import logging
import random
import numpy as np
import urllib.request
import tiktoken
from typing import Any, Dict, List, Literal, Tuple

try:
    import torch
except ImportError as exc:
    raise RuntimeError("PyTorch is required. Install torch before running this script.") from exc

from .config import TrainingConfig
from .common import ensure_dir, utc_now
from .core import (
    TINY_SHAKESPEARE_URL,
    WIKITEXT2_TRAIN_URL,
    WIKITEXT2_VALID_URL,
    WIKITEXT2_TEST_URL,
)


class CharTokenizer:
    def __init__(self, text: str):
        vocab = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(vocab)
        self.merges: List[Tuple[str, str]] = []

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def update_merges(self, merges: List[Tuple[str, str]]) -> None:
        normalized: List[Tuple[str, str]] = []
        for item in merges:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Each merge entry must be a 2-tuple.")
            left, right = item
            normalized.append((str(left), str(right)))
        self.merges = normalized


class TinyShakespeareData:
    def __init__(self, data_dir: str, cfg: TrainingConfig, logger: logging.Logger):
        self.data_dir = ensure_dir(data_dir)
        self.cfg = cfg
        self.logger = logger
        self.input_path = self.data_dir / "input.txt"
        if not self.input_path.exists():
            self.logger.info("Downloading Tiny Shakespeare to %s", self.input_path)
            urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, self.input_path)
        self.text = self.input_path.read_text(encoding="utf-8")
        split_idx = int(len(self.text) * cfg.train_val_split)
        self.train_text = self.text[:split_idx]
        self.val_text = self.text[split_idx:]
        self.tokenizer = CharTokenizer(self.text)
        self.train_ids = torch.tensor(self.tokenizer.encode(self.train_text), dtype=torch.long)
        self.val_ids = torch.tensor(self.tokenizer.encode(self.val_text), dtype=torch.long)
        self.augmentation_history: List[Dict[str, Any]] = []

    def rebuild_after_tokenizer_update(self) -> None:
        self.train_ids = torch.tensor(self.tokenizer.encode(self.train_text), dtype=torch.long)
        self.val_ids = torch.tensor(self.tokenizer.encode(self.val_text), dtype=torch.long)

    def apply_augmentation(self, intensity: float, pattern: str) -> Dict[str, Any]:
        intensity = max(0.0, min(float(intensity), self.cfg.max_augmented_fraction))
        n_chars = int(len(self.train_text) * intensity)
        if n_chars == 0:
            return {"changed_chars": 0, "pattern": pattern}
        rng = random.Random(self.cfg.seed + len(self.augmentation_history))
        text_list = list(self.train_text)
        indices = rng.sample(range(len(text_list)), k=min(n_chars, len(text_list)))
        substitutions = {
            "a": "@",
            "e": "3",
            "i": "1",
            "o": "0",
            "s": "$",
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
        self.train_text = "".join(text_list)
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


class WikiText2Data:
    def __init__(self, data_dir: str, cfg: TrainingConfig, logger: logging.Logger):
        self.data_dir = ensure_dir(data_dir)
        self.cfg = cfg
        self.logger = logger
        self.train_path = self.data_dir / "wiki.train.raw"
        self.valid_path = self.data_dir / "wiki.valid.raw"
        self.test_path = self.data_dir / "wiki.test.raw"
        self._prepare_dataset()
        self.train_text = self.train_path.read_text(encoding="utf-8")
        self.val_text = self.valid_path.read_text(encoding="utf-8")
        self.test_text = self.test_path.read_text(encoding="utf-8")
        vocab_text = self.train_text + self.val_text + self.test_text
        self.tokenizer = CharTokenizer(vocab_text)
        self.text = self.train_text
        self.train_ids = torch.tensor(self.tokenizer.encode(self.train_text), dtype=torch.long)
        self.val_ids = torch.tensor(self.tokenizer.encode(self.val_text), dtype=torch.long)
        self.augmentation_history: List[Dict[str, Any]] = []

    def _prepare_dataset(self) -> None:
        downloads = [
            (self.train_path, WIKITEXT2_TRAIN_URL),
            (self.valid_path, WIKITEXT2_VALID_URL),
            (self.test_path, WIKITEXT2_TEST_URL),
        ]
        for path, url in downloads:
            if not path.exists():
                self.logger.info("Downloading WikiText-2 raw file to %s", path)
                urllib.request.urlretrieve(url, path)
        if self.train_path.stat().st_size < 1000000:
            raise ValueError("WikiText-2 train split appears incomplete or corrupted.")
        if self.valid_path.stat().st_size < 100000:
            raise ValueError("WikiText-2 validation split appears incomplete or corrupted.")
        if self.test_path.stat().st_size < 100000:
            raise ValueError("WikiText-2 test split appears incomplete or corrupted.")

    def rebuild_after_tokenizer_update(self) -> None:
        self.text = self.train_text
        self.train_ids = torch.tensor(self.tokenizer.encode(self.train_text), dtype=torch.long)
        self.val_ids = torch.tensor(self.tokenizer.encode(self.val_text), dtype=torch.long)

    def apply_augmentation(self, intensity: float, pattern: str) -> Dict[str, Any]:
        intensity = max(0.0, min(float(intensity), self.cfg.max_augmented_fraction))
        n_chars = int(len(self.train_text) * intensity)
        if n_chars == 0:
            return {"changed_chars": 0, "pattern": pattern}
        rng = random.Random(self.cfg.seed + len(self.augmentation_history))
        text_list = list(self.train_text)
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
        self.train_text = "".join(text_list)
        self.text = self.train_text
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



class GPT2TokenizerShim:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab
        self.merges: List[Tuple[str, str]] = []

    def encode(self, text: str) -> List[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids: List[int]) -> str:
        normalized = [int(i) for i in ids if int(i) >= 0 and int(i) < self.vocab_size]
        return self.enc.decode(normalized)

    def update_merges(self, merges: List[Tuple[str, str]]) -> None:
        normalized: List[Tuple[str, str]] = []
        for item in merges:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Each merge entry must be a 2-tuple.")
            left, right = item
            normalized.append((str(left), str(right)))
        self.merges = normalized


class OpenWebTextData:
    def __init__(self, data_dir: str, cfg: TrainingConfig, logger: logging.Logger):
        self.data_dir = ensure_dir(data_dir)
        self.cfg = cfg
        self.logger = logger

        self.train_path = self.data_dir / "train.bin"
        self.val_path = self.data_dir / "val.bin"
        self.meta_path = self.data_dir / "meta.json"

        if not self.train_path.exists():
            raise FileNotFoundError(f"Missing OpenWebText train.bin at {self.train_path}")
        if not self.val_path.exists():
            raise FileNotFoundError(f"Missing OpenWebText val.bin at {self.val_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing OpenWebText meta.json at {self.meta_path}")

        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.tokenizer = GPT2TokenizerShim()

        self.train_ids = np.memmap(self.train_path, dtype=np.uint16, mode="r")
        self.val_ids = np.memmap(self.val_path, dtype=np.uint16, mode="r")

        self.text = json.dumps(self.meta, sort_keys=True)
        self.train_text = self.text
        self.val_text = self.text
        self.augmentation_history: List[Dict[str, Any]] = []

    def rebuild_after_tokenizer_update(self) -> None:
        return None

    def apply_augmentation(self, intensity: float, pattern: str) -> Dict[str, Any]:
        return {"changed_chars": 0, "pattern": pattern, "skipped": True, "reason": "openwebtext_bin_dataset"}

    def get_batch(self, split: Literal["train", "val"], batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_ids if split == "train" else self.val_ids
        data_len = len(data)
        if data_len <= block_size + 1:
            raise ValueError("Dataset is too small for the configured block size.")

        starts = torch.randint(data_len - block_size - 1, (batch_size,))
        x_list: List[torch.Tensor] = []
        y_list: List[torch.Tensor] = []

        for i in starts.tolist():
            x_np = np.array(data[i:i + block_size], dtype=np.int64)
            y_np = np.array(data[i + 1:i + 1 + block_size], dtype=np.int64)
            x_list.append(torch.from_numpy(x_np).long())
            y_list.append(torch.from_numpy(y_np).long())

        x = torch.stack(x_list)
        y = torch.stack(y_list)

        if device.startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)
def build_dataset(dataset_name: str, data_dir: str, cfg: TrainingConfig, logger: logging.Logger):
    if dataset_name == "tinyshakespeare":
        return TinyShakespeareData(data_dir, cfg, logger)
    if dataset_name == "wikitext2":
        return WikiText2Data(data_dir, cfg, logger)
    if dataset_name == "openwebtext":
        return OpenWebTextData(data_dir, cfg, logger)
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")
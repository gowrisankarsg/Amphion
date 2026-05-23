#!/usr/bin/env python3
"""
train_maskgct_t2s.py  —  Final v4
Continuous-learning T2S trainer: Tamil + EN + ZH.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUG FIXES (v3 → v4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[B1] verify_ta_phone_offset no longer calls dataset.__getitem__().
     Reads .npy directly to avoid polluting bad_indices with
     transient spot-check I/O errors.
[B2] EWC Fisher & reference tensors moved to device once at
     construction. penalty() no longer calls .to(device) per step.
[B3] torch.load uses weights_only=True (PyTorch 2.6+ safe).
[B4] End-of-epoch gradient flush added: optimizer.zero_grad()
     after epoch loop to prevent gradient bleed across epochs.

NEW FEATURES (v3 → v4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[F1] --root-dir: base directory for resolving codes/phones .npy
     paths from manifest. Manifest stores relative paths only
     (e.g. "codes/ta_0001.npy"). root-dir/language/ is prepended.
[F2] --output-dir: directory for all checkpoints and logs.
[F3] best_val_loss saved in checkpoint and restored on resume.
[F4] latest.pth removed — no symlink created.

H100 / H200 LARGE-BATCH NOTES (batch_size 200–400+)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- At batch_size=300: 300 × ~400 semantic frames × 1024 hidden = ~120M
  float16 values per forward → ~240MB activation memory per batch.
  With grad_accumulation=1, this is fine on H100 80GB.
- Loss is already mean-reduced inside _masked_ce_loss → no LR rescaling
  needed for large batches (loss magnitude stays constant).
- grad_clip=1.0 remains correct regardless of batch size.
- For batch_size > 256, consider --warmup-steps 1000+ to stabilize
  early large-gradient steps on phone_emb Tamil rows (363-370).
- With num_workers=8+, pin_memory=True is critical on H100/H200 for
  full PCIe/NVLink bandwidth utilization.

MANIFEST FORMAT (one JSON object per line):
  {
    "id":                  "ta_0001",
    "semantic_codes_path": "codes/ta_0001.npy",   ← relative to root-dir/language/
    "phone_ids_path":      "phones/ta_0001.npy",  ← relative to root-dir/language/
    "language":            "ta",
    "code_len":            412,
    "phone_len":           38
  }

DIRECTORY LAYOUT expected under --root-dir:
  root-dir/
    ta/
      codes/   *.npy    int32 (T,)  semantic tokens range [0, 8191]
      phones/  *.npy    int32 (L,)  phone IDs range [0, 1022]
    en/
      codes/   *.npy
      phones/  *.npy
    zh/
      codes/   *.npy
      phones/  *.npy

PHONE ID ASSIGNMENT:
  EN/ZH/JA/KO/FR/DE : IDs   0–362  (existing vocab.json, 363 entries)
  Tamil new phones  : IDs 363–370  (ʉ ʈ ɻ ɖ ɭ ɳ ʂ ɣ — 8 phonemes)
  Free slots        : IDs 371–1022 (future languages)
  Padding token     : ID  1023     (nn.Embedding padding_idx, never in .npy)
  Embedding table   : 1024 slots   (no expansion needed)

EXAMPLE RUN:
  python train_maskgct_t2s.py \
    --root-dir        /data/maskgct_processed \
    --output-dir      /checkpoints/t2s_tamil_v1 \
    --train-manifest  manifests/ta_train.jsonl::ta \
    --train-manifest  manifests/en_train.jsonl::en \
    --train-manifest  manifests/zh_train.jsonl::zh \
    --val-manifest    manifests/ta_val.jsonl::ta \
    --val-manifest    manifests/en_val.jsonl::en \
    --val-manifest    manifests/zh_val.jsonl::zh \
    --base-checkpoint checkpoints/t2s_model/model.safetensors \
    --config          models/tts/maskgct/config/maskgct.json \
    --batch-size 300 --grad-accumulation 1 --epochs 30 \
    --lang-balance ta:6,en:2,zh:2 \
    --ewc-lambda 5000 --ewc-fisher-batches 200 \
    --warmup-steps 1000 --num-workers 8 \
    --amp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import datetime
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup
import safetensors.torch

from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
from utils.util import load_config


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MaskGCT T2S continual-learning finetuning (Tamil + EN + ZH)."
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    # [F1] root-dir: base directory for all .npy files referenced in manifests.
    #   Layout:  root-dir/<language>/codes/*.npy
    #            root-dir/<language>/phones/*.npy
    #   Manifest stores RELATIVE paths: "codes/ta_0001.npy"
    #   Resolved as: root-dir / language / "codes/ta_0001.npy"
    p.add_argument(
        "--root-dir", type=Path, required=True,
        help="Root directory containing language sub-folders with codes/ and phones/ "
             "sub-directories. Manifest paths are resolved as: "
             "root-dir/<language>/<manifest_path>."
    )
    # [F2] output-dir: all checkpoints and TensorBoard logs are saved here.
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for saving checkpoints (step_*.pth, best_model.pth) "
             "and TensorBoard logs."
    )

    # ── Manifests ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--train-manifest", dest="train_manifests", action="append", required=True,
        help="Training manifest JSONL, suffixed with '::lang'. "
             "Repeat for each language. Example: ta_train.jsonl::ta"
    )
    p.add_argument(
        "--val-manifest", dest="val_manifests", action="append", required=True,
        help="Validation manifest JSONL, suffixed with '::lang'. "
             "Same suffix convention as --train-manifest."
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--config", type=Path,
        default=Path("models/tts/maskgct/config/maskgct.json")
    )
    p.add_argument(
        "--base-checkpoint", type=Path, required=True,
        help="Pretrained T2S safetensors checkpoint "
             "(e.g. from huggingface amphion/MaskGCT)."
    )

    # ── Phone offset ──────────────────────────────────────────────────────────
    # NOT used in the forward pass — only for startup verify_ta_phone_offset().
    # See module docstring for full phone ID assignment table.
    p.add_argument(
        "--ta-phone-offset", type=int, default=363,
        help="[Preprocessing contract — not used in forward pass] "
             "First phone ID assigned to Tamil-specific phonemes in pre-extracted "
             ".npy files. Must match your G2P preprocessing script. Default=363."
    )

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--batch-size",        type=int,   default=32,
                   help="Per-GPU batch size. H100 80GB supports 200–400+ at bf16/fp16.")
    p.add_argument("--grad-accumulation", type=int,   default=4,
                   help="Gradient accumulation steps. "
                        "Effective batch = batch_size × grad_accumulation.")
    p.add_argument("--epochs",            type=int,   default=30)
    p.add_argument("--max-steps",         type=int,   default=0,
                   help="Hard cap on optimizer steps. 0 = run all epochs.")
    p.add_argument("--learning-rate",     type=float, default=2e-5,
                   help="LR for phone_emb (Tamil rows). "
                        "Backbone LR = lr × backbone-lr-scale.")
    p.add_argument("--backbone-lr-scale", type=float, default=0.1,
                   help="Backbone LR multiplier vs phone_emb LR.")
    p.add_argument("--weight-decay",      type=float, default=0.01)
    p.add_argument("--warmup-steps",      type=int,   default=500,
                   help="LR warmup steps. Use 1000+ for batch_size > 256.")
    p.add_argument("--grad-clip",         type=float, default=1.0)
    p.add_argument(
        "--amp", action="store_true",
        help="Enable CUDA AMP (bfloat16 on H100/H200, float16 fallback). "
             "Strongly recommended for large batches."
    )
    p.add_argument("--num-workers",       type=int,   default=4,
                   help="DataLoader workers. Use 8–16 on H100/H200 with fast NVMe.")
    p.add_argument("--seed",              type=int,   default=42)

    # ── Language balancing ────────────────────────────────────────────────────
    p.add_argument(
        "--lang-balance", type=str, default="ta:6,en:2,zh:2",
        help="Sampling weight ratio per language. "
             "'ta:6,en:2,zh:2' → Tamil 60%% of batches."
    )

    # ── EWC ───────────────────────────────────────────────────────────────────
    p.add_argument("--ewc-lambda",        type=float, default=0.0,
                   help="EWC regularisation strength. 0 = disabled. Try 5000–50000.")
    p.add_argument(
        "--ewc-fisher-batches", type=int, default=200,
        help="EN/ZH *batches* (not samples) for Fisher diagonal estimation. "
             "Effective samples = ewc_fisher_batches × batch_size."
    )

    # ── Logging / checkpointing ───────────────────────────────────────────────
    p.add_argument("--log-interval",  type=int, default=50,
                   help="TensorBoard log every N optimizer steps.")
    p.add_argument("--val-interval",  type=int, default=500,
                   help="Validate every N optimizer steps. 0 = once per epoch.")
    p.add_argument("--save-every",    type=int, default=1000,
                   help="Save checkpoint every N optimizer steps.")
    p.add_argument("--keep-last",     type=int, default=3,
                   help="Keep only the last N step checkpoints. "
                        "best_model.pth is never pruned.")
    p.add_argument(
        "--resume", type=str, default="",
        help="Path to a .pth checkpoint to resume from. "
             "Restores model, optimizer, scheduler, scaler, step, epoch, "
             "and best_val_loss."
    )

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Manifest parsing
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ManifestSpec:
    path: Path
    language: Optional[str] = None


def parse_manifest_specs(
    entries: Sequence[str], flag_name: str
) -> List[ManifestSpec]:
    """
    Parses 'path/to/manifest.jsonl::lang' entries.
    '::lang' suffix is optional but required for correct language tagging.
    """
    if not entries:
        raise ValueError(f"{flag_name} requires at least one manifest.")
    specs: List[ManifestSpec] = []
    for raw in entries:
        val = raw.strip()
        lang: Optional[str] = None
        if "::" in val:
            path_str, lang_part = val.rsplit("::", 1)
            val  = path_str.strip()
            lang = lang_part.strip().lower() or None
        specs.append(ManifestSpec(path=Path(val).expanduser(), language=lang))
    return specs


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class T2SSample:
    id:                   str
    semantic_codes_path:  Path   # int32 .npy  (T,)  semantic tokens [0, 8191]
    phone_ids_path:       Path   # int32 .npy  (L,)  phone IDs       [0, 1022]
    language:             str
    code_len:             int
    phone_len:            int


class T2SDataset(Dataset):
    """
    Loads pre-extracted semantic codes (.npy) and phone ID sequences (.npy).

    Path resolution:
      Manifest stores relative paths: "codes/ta_0001.npy"
      Full path = root_dir / language / manifest_path
        e.g.  /data/maskgct_processed / ta / codes/ta_0001.npy

    CFG dropout is handled by MaskGCT_T2S.forward_diffusion() internally:
      85% of batches → real prompt prefix (0–40% of sequence length)
      15% of batches → prompt_len = 0  (null conditioning for CFG training)
    No paired reference clip is needed here — only for inference.
    """

    def __init__(
        self,
        specs: Sequence[ManifestSpec],
        root_dir: Path,
    ) -> None:
        self.root_dir = root_dir.resolve()
        self.samples: List[T2SSample]          = []
        self.bad_indices: Set[int]             = set()
        self.lang_to_indices: Dict[str, List[int]] = defaultdict(list)

        for spec in specs:
            self._load_manifest(spec)

        # [C7] Warn on unknown-language samples
        unknown = self.lang_to_indices.get("unknown", [])
        if unknown:
            print(
                f"\n[Dataset] ⚠ WARNING: {len(unknown)} samples have "
                f"language='unknown'.\n"
                f"  These receive near-zero sampling weight and corrupt EWC Fisher.\n"
                f"  Fix: add '::lang' suffix to --train-manifest / --val-manifest.\n"
                f"  Example: manifests/ta_train.jsonl::ta\n"
            )

        if not self.samples:
            raise RuntimeError(
                "No samples loaded. Check --root-dir layout, manifest paths, "
                "and field names."
            )

        print(f"\n[Dataset] Loaded {len(self.samples):,} total samples:")
        for lang, idxs in sorted(self.lang_to_indices.items()):
            print(f"  [{lang:>10}]  {len(idxs):>8,} samples")

    def _load_manifest(self, spec: ManifestSpec) -> None:
        if not spec.path.exists():
            raise FileNotFoundError(f"Manifest not found: {spec.path}")

        loaded = skipped = 0
        with spec.path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  [Skip] {spec.path.name} line {lineno}: {e}")
                    skipped += 1
                    continue

                # Accept both field name variants for flexibility
                codes_rel  = (rec.get("semantic_codes_path")
                              or rec.get("codes_path", ""))
                phones_rel = (rec.get("phone_ids_path")
                              or rec.get("text_ids_path", ""))
                lang       = (rec.get("language") or spec.language or "unknown").lower()

                if not codes_rel or not phones_rel:
                    print(
                        f"  [Skip] {spec.path.name} line {lineno}: "
                        f"missing semantic_codes_path or phone_ids_path."
                    )
                    skipped += 1
                    continue

                # [F1] Resolve: root_dir / language / relative_path_from_manifest
                codes_path  = self.root_dir / lang / codes_rel
                phones_path = self.root_dir / lang / phones_rel

                idx = len(self.samples)
                self.samples.append(T2SSample(
                    id=rec.get("id", f"sample_{idx}"),
                    semantic_codes_path=codes_path,
                    phone_ids_path=phones_path,
                    language=lang,
                    code_len=int(rec.get("code_len", 0)),
                    phone_len=int(rec.get("phone_len", 0)),
                ))
                self.lang_to_indices[lang].append(idx)
                loaded += 1

        print(
            f"  {spec.path.name} "
            f"(lang={spec.language or 'from_field'}): "
            f"loaded={loaded:,}, skipped={skipped}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        if idx in self.bad_indices:
            return None

        sample = self.samples[idx]
        try:
            codes  = np.load(sample.semantic_codes_path,
                             allow_pickle=False).astype(np.int64)
            phones = np.load(sample.phone_ids_path,
                             allow_pickle=False).astype(np.int64)

            if codes.size == 0 or phones.size == 0:
                raise ValueError("Empty .npy file")
            if codes.max() > 8191:
                raise ValueError(
                    f"Semantic code {codes.max()} out of range [0, 8191]"
                )
            if phones.max() > 1022:
                # ID 1023 = PAD — should never appear in stored phone files
                raise ValueError(
                    f"Phone ID {phones.max()} out of range [0, 1022] "
                    f"(1023 is PAD, never stored in .npy)"
                )

            code_len  = int(codes.shape[0])
            phone_len = int(phones.shape[0])

            if sample.code_len  == 0: sample.code_len  = code_len
            if sample.phone_len == 0: sample.phone_len = phone_len

            return {
                "id":        sample.id,
                "codes":     torch.from_numpy(codes),   # (T,) int64
                "phones":    torch.from_numpy(phones),  # (L,) int64
                "language":  sample.language,
                "code_len":  code_len,
                "phone_len": phone_len,
            }

        except Exception as e:
            if idx not in self.bad_indices:
                print(f"[Warn] Skipping '{sample.id}': {e}")
                self.bad_indices.add(idx)
            return None


def collate_t2s(batch: List[Optional[Dict]]) -> Optional[Dict]:
    """
    Pads codes and phones to max length in the batch.

    x_mask     : 1 = valid semantic frame, 0 = padding
    phone_mask : 1 = valid phone token,    0 = padding
    phones padded with 1023 (phone_emb padding_idx → zero embedding, no grad)
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    codes_padded  = pad_sequence(
        [b["codes"]  for b in batch], batch_first=True, padding_value=0
    )
    phones_padded = pad_sequence(
        [b["phones"] for b in batch], batch_first=True, padding_value=1023
    )

    B = len(batch)
    x_mask     = torch.zeros(B, codes_padded.shape[1],  dtype=torch.float32)
    phone_mask = torch.zeros(B, phones_padded.shape[1], dtype=torch.float32)

    for i, b in enumerate(batch):
        x_mask[i,     :b["code_len"]]  = 1.0
        phone_mask[i, :b["phone_len"]] = 1.0

    return {
        "codes":      codes_padded,
        "phones":     phones_padded,
        "x_mask":     x_mask,
        "phone_mask": phone_mask,
        "languages":  [b["language"] for b in batch],
        "ids":        [b["id"]       for b in batch],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Language-balanced sampler
# ══════════════════════════════════════════════════════════════════════════════

def parse_lang_balance(spec: str) -> Dict[str, float]:
    """'ta:6,en:2,zh:2' → {'ta': 6.0, 'en': 2.0, 'zh': 2.0}"""
    result: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lang, _, wt = part.partition(":")
        result[lang.strip()] = float(wt.strip()) if wt.strip() else 1.0
    return result


def build_weighted_sampler(
    dataset: T2SDataset,
    lang_weights: Dict[str, float],
) -> WeightedRandomSampler:
    """
    Per-sample weight = lang_weight / lang_sample_count.

    Example: ta:6, en:2, zh:2 with 10k ta / 3k en / 3k zh samples:
      w_ta = 6/10000 = 0.0006   → Tamil  = 6/(6+2+2) = 60% of draws
      w_en = 2/3000  = 0.00067  → EN     = 20% of draws
      w_zh = 2/3000  = 0.00067  → ZH     = 20% of draws
    EN/ZH are over-sampled relative to data size to prevent forgetting.
    """
    per_lang_w: Dict[str, float] = {}
    for lang, weight in lang_weights.items():
        count = len(dataset.lang_to_indices.get(lang, []))
        if count == 0:
            print(f"[Sampler] ⚠ '{lang}' has 0 samples — skipped in balance.")
            continue
        per_lang_w[lang] = weight / count

    sample_weights = [
        per_lang_w.get(s.language, 1e-9)
        for s in dataset.samples
    ]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Model builder
# ══════════════════════════════════════════════════════════════════════════════

def build_t2s_model(
    cfg_path: Path,
    base_checkpoint: Path,
    device: torch.device,
) -> MaskGCT_T2S:
    """
    No embedding table expansion needed:
      phone_emb = Embedding(1024, 1024, padding_idx=1023)
      Slots 363–370 are free and initialised at construction with N(0, 0.02).
      Tamil phonemes (ʉ ʈ ɻ ɖ ɭ ɳ ʂ ɣ) are assigned IDs 363–370 in preprocessing.
    """
    cfg    = load_config(str(cfg_path))
    t2s_c  = cfg.model.t2s_model
    model  = MaskGCT_T2S(cfg=t2s_c)

    print(f"[Model] Loading: {base_checkpoint}")
    missing, unexpected = safetensors.torch.load_model(
        model, str(base_checkpoint), strict=False
    )
    if missing:
        print(f"  Missing   ({len(missing)}): {missing[:5]}"
              f"{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected({len(unexpected)}): {unexpected[:5]}")

    model = model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {total/1e6:.1f}M total | {trainable/1e6:.1f}M trainable")
    print(
        f"[Model] phone_emb: 1024 slots | "
        f"0–362 EN/ZH/JA/KO/FR/DE | "
        f"363–370 Tamil (ʉ ʈ ɻ ɖ ɭ ɳ ʂ ɣ) | "
        f"371–1022 free | 1023 PAD"
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# EWC
# ══════════════════════════════════════════════════════════════════════════════

class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al. 2017).

    L_total = L_ce + (λ/2) * Σ_i  F_i * (θ_i − θ*_i)²

    [B2] Fisher tensors and reference (θ*) tensors are moved to device ONCE
         at construction. penalty() does zero .to() calls — important at
         batch_size 300+ where penalty() runs every optimizer step.

    [R1] model.train() is intentionally used for Fisher estimation.
         MaskGCT_T2S has zero Dropout/BatchNorm layers.
         train() is architecturally correct: Fisher must be estimated under
         the same stochastic masking distribution as the training loss.

    [R2] n_batches: Fisher is divided by batch count (not sample count).
         DataLoader loss is already mean-reduced within each batch.
         Each batch contributes exactly one squared-gradient sample.
    """

    def __init__(
        self,
        model:             MaskGCT_T2S,
        reference_loader:  DataLoader,
        device:            torch.device,
        lambda_:           float = 5000.0,
        n_batches:         int   = 200,
    ) -> None:
        self.lambda_ = lambda_
        self.device  = device

        # θ* — pretrained weights frozen to device [B2]
        self._params: Dict[str, torch.Tensor] = {
            name: param.clone().detach().to(device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        print(
            f"[EWC] Estimating Fisher diagonal "
            f"({n_batches} EN/ZH batches)..."
        )
        # Fisher stored on device from construction [B2]
        self._fisher = self._compute_fisher(model, reference_loader, n_batches)
        print(
            f"[EWC] Ready. λ={lambda_:.0f} | "
            f"Tracked params: {len(self._fisher)}"
        )

    def _compute_fisher(
        self,
        model:     MaskGCT_T2S,
        loader:    DataLoader,
        n_batches: int,
    ) -> Dict[str, torch.Tensor]:
        # [B2] Initialise on device directly
        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # [R1] model.train() intentional — see class docstring
        model.train()
        batch_count = 0

        for batch in loader:
            if batch is None or batch_count >= n_batches:
                break

            codes      = batch["codes"].to(self.device)
            phones     = batch["phones"].to(self.device)
            x_mask     = batch["x_mask"].to(self.device)
            phone_mask = batch["phone_mask"].to(self.device)

            model.zero_grad()
            logits, final_mask, x0, _, _ = model(
                codes, x_mask, phone_id=phones, phone_mask=phone_mask
            )
            loss = _masked_ce_loss(logits, final_mask, x0)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

            batch_count += 1

        # [R2] Divide by batch count
        for name in fisher:
            fisher[name] /= max(batch_count, 1)

        model.zero_grad()
        print(f"[EWC] Fisher estimated over {batch_count} batches "
              f"(requested {n_batches}).")
        return fisher  # already on device [B2]

    def penalty(self, model: MaskGCT_T2S) -> torch.Tensor:
        """
        L_ewc = (λ/2) * Σ_i  F_i * (θ_i − θ*_i)²
        [B2] No .to(device) calls — Fisher and θ* are already on device.
        """
        loss = torch.tensor(0.0, device=self.device)
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self._fisher:
                continue
            loss = loss + (self._fisher[name] * (param - self._params[name]).pow(2)).sum()
        return (self.lambda_ / 2.0) * loss


# ══════════════════════════════════════════════════════════════════════════════
# Loss helpers
# ══════════════════════════════════════════════════════════════════════════════

def _masked_ce_loss(
    logits:     torch.Tensor,   # (B, T, 8192)
    final_mask: torch.Tensor,   # (B, T, 1)   1 = masked position → predict
    x0:         torch.Tensor,   # (B, T)      ground-truth semantic codes
) -> torch.Tensor:
    """
    L = -Σ_i  m_i * log p_θ(x_i | X_t, C)
    Only masked positions (final_mask=1) contribute.
    final_mask = forward_diffusion_mask AND x_mask → padding always excluded.
    Mean-reduced: loss magnitude is independent of batch size and sequence length.
    """
    B, T, V   = logits.shape
    logits_f  = logits.reshape(B * T, V)
    targets_f = x0.reshape(B * T)
    mask_f    = final_mask.reshape(B * T)
    ce_all    = F.cross_entropy(logits_f, targets_f, reduction="none")
    return (ce_all * mask_f).sum() / mask_f.sum().clamp(min=1.0)


@torch.no_grad()
def compute_accuracy(
    logits:     torch.Tensor,
    final_mask: torch.Tensor,
    x0:         torch.Tensor,
) -> float:
    """Top-1 accuracy over masked (to-predict) positions."""
    preds   = logits.argmax(dim=-1)
    mask    = final_mask.squeeze(-1)
    valid   = mask.sum().item()
    return ((preds == x0).float() * mask).sum().item() / valid if valid > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Startup validator  [B1 fix — never calls dataset.__getitem__]
# ══════════════════════════════════════════════════════════════════════════════

def verify_ta_phone_offset(
    dataset:         T2SDataset,
    ta_phone_offset: int,
    n_check:         int = 50,
) -> None:
    """
    Spot-checks Tamil samples by reading .npy files DIRECTLY — never via
    dataset.__getitem__() — so transient I/O failures here cannot add any
    index to dataset.bad_indices and permanently blacklist training samples.

    Confirms that at least one Tamil phone ID >= ta_phone_offset, verifying
    that preprocessing assigned Tamil-specific phonemes (ʉ ʈ ɻ ɖ ɭ ɳ ʂ ɣ)
    to slots 363–370 rather than colliding with EN/ZH IDs 0–362.
    """
    ta_indices = dataset.lang_to_indices.get("ta", [])
    if not ta_indices:
        print("[Verify] No Tamil samples — skipping ta-phone-offset check.")
        return

    sample_indices = random.sample(ta_indices, min(n_check, len(ta_indices)))
    max_id_seen    = 0
    checked        = 0

    for idx in sample_indices:
        sample = dataset.samples[idx]
        try:
            # [B1] Direct numpy load — NOT dataset[idx] — no bad_indices pollution
            phones = np.load(sample.phone_ids_path, allow_pickle=False)
            if phones.size > 0:
                max_id_seen = max(max_id_seen, int(phones.max()))
            checked += 1
        except Exception as e:
            # Silently skip — this is a startup check, not training
            print(f"  [Verify] Could not read {sample.phone_ids_path.name}: {e}")

    if checked == 0:
        print("[Verify] ⚠ All spot-check Tamil samples failed to read.")
        return

    if max_id_seen < ta_phone_offset:
        print(
            f"\n[Verify] ⚠ WARNING: --ta-phone-offset={ta_phone_offset} but max "
            f"Tamil phone ID in {checked} samples = {max_id_seen}.\n"
            f"  Expected at least one ID >= {ta_phone_offset} for Tamil phonemes "
            f"(ʉ ʈ ɻ ɖ ɭ ɳ ʂ ɣ).\n"
            f"  Likely cause: G2P preprocessing did not assign Tamil-specific "
            f"phones starting at ID {ta_phone_offset}.\n"
        )
    else:
        print(
            f"[Verify] ✓ ta-phone-offset OK: "
            f"max Tamil phone ID={max_id_seen} >= {ta_phone_offset} "
            f"({checked} samples checked)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Checkpointing  [F3 — best_val_loss in checkpoint]  [F4 — no latest.pth]
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    path:               Path,
    model:              nn.Module,
    optimizer:          torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch:              int,
    step:               int,
    best_val_loss:      float,          # [F3]
    recent_checkpoints: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model":              model.state_dict(),
            "optimizer":          optimizer.state_dict(),
            "scheduler":          scheduler.state_dict() if scheduler else None,
            "scaler":             scaler.state_dict()    if scaler    else None,
            "epoch":              epoch,
            "step":               step,
            "best_val_loss":      best_val_loss,          # [F3]
            "recent_checkpoints": recent_checkpoints,
        },
        path,
    )


def prune_old_checkpoints(recent: List[str], keep_last: int) -> List[str]:
    """Remove oldest step checkpoints beyond keep_last. best_model.pth is not in this list."""
    while len(recent) > keep_last:
        old = recent.pop(0)
        try:
            if os.path.exists(old):
                os.remove(old)
                print(f"[Ckpt] Pruned: {Path(old).name}")
        except OSError as e:
            print(f"[Ckpt] Could not remove {old}: {e}")
    return recent


# ══════════════════════════════════════════════════════════════════════════════
# Validation  [R3 — per-lang loss re-indexed to language rows]
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model:  MaskGCT_T2S,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Training-time validation: single forward pass at random t ~ U[1e-5, 1.0].
    This is NOT inference. For real quality evaluation use model.reverse_diffusion()
    with a reference audio clip.

    [R3] Per-language CE is computed by re-indexing logits/mask/x0 to only
    the rows of that language. Mixed-batch mean is never credited to a language.
    """
    model.eval()
    total_loss = total_acc = 0.0
    count      = 0
    lang_loss:  Dict[str, float] = defaultdict(float)
    lang_count: Dict[str, int]   = defaultdict(int)

    for batch in loader:
        if batch is None:
            continue

        codes      = batch["codes"].to(device)
        phones     = batch["phones"].to(device)
        x_mask     = batch["x_mask"].to(device)
        phone_mask = batch["phone_mask"].to(device)
        languages  = batch["languages"]

        logits, final_mask, x0, _, _ = model(
            codes, x_mask, phone_id=phones, phone_mask=phone_mask
        )
        loss = _masked_ce_loss(logits, final_mask, x0).item()
        acc  = compute_accuracy(logits, final_mask, x0)
        bsz  = codes.shape[0]

        total_loss += loss * bsz
        total_acc  += acc  * bsz
        count      += bsz

        # [R3] Per-language: recompute on language-specific rows only
        for lang in set(languages):
            idx_list = [i for i, l in enumerate(languages) if l == lang]
            if not idx_list:
                continue
            idx_t   = torch.tensor(idx_list, device=device)
            lang_ce = _masked_ce_loss(
                logits[idx_t], final_mask[idx_t], x0[idx_t]
            ).item()
            lang_loss[lang]  += lang_ce * len(idx_list)
            lang_count[lang] += len(idx_list)

    model.train()

    if count == 0:
        return {"loss": 0.0, "top1_acc": 0.0}

    results: Dict[str, float] = {
        "loss":     total_loss / count,
        "top1_acc": total_acc  / count,
    }
    for lang in sorted(lang_loss):
        results[f"loss_{lang}"] = lang_loss[lang] / max(lang_count[lang], 1)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Main] Device   : {device}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    print(f"[Main] PyTorch  : {torch.__version__}")
    print(f"[Main] root-dir : {args.root_dir.resolve()}")
    print(f"[Main] output-dir: {args.output_dir.resolve()}")
    print(f"[Main] PID      : {os.getpid()}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = os.environ.get(
        "T2S_RUN_NAME",
        f"t2s_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    writer = SummaryWriter(log_dir=str(output_dir / "logs" / run_name))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_t2s_model(args.config, args.base_checkpoint, device)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_specs = parse_manifest_specs(args.train_manifests, "--train-manifest")
    val_specs   = parse_manifest_specs(args.val_manifests,   "--val-manifest")

    # [F1] root_dir passed to dataset — paths resolved as root_dir/lang/rel_path
    train_dataset = T2SDataset(train_specs, root_dir=args.root_dir)
    val_dataset   = T2SDataset(val_specs,   root_dir=args.root_dir)

    # [B1] verify reads .npy directly — never touches bad_indices
    verify_ta_phone_offset(train_dataset, args.ta_phone_offset)

    lang_weights  = parse_lang_balance(args.lang_balance)
    train_sampler = build_weighted_sampler(train_dataset, lang_weights)

    # H100/H200: pin_memory=True is critical for NVLink/PCIe bandwidth
    # persistent_workers=True avoids worker respawn overhead at large batch sizes
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_t2s,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_t2s,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ── Optimizer: split LR groups ────────────────────────────────────────────
    # phone_emb (Tamil rows 363–370) need higher LR to learn new phonemes fast
    # backbone (DiffLlamaPrefix, 16 Llama layers) uses lower LR to prevent forgetting
    backbone_params  = [p for n, p in model.named_parameters()
                        if p.requires_grad and "phone_emb" not in n]
    phone_emb_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and "phone_emb" in n]

    print(
        f"[Optim] backbone  : {sum(p.numel() for p in backbone_params)/1e6:.1f}M  "
        f"lr={args.learning_rate * args.backbone_lr_scale:.2e}"
    )
    print(
        f"[Optim] phone_emb : {sum(p.numel() for p in phone_emb_params)/1e6:.3f}M  "
        f"lr={args.learning_rate:.2e}"
    )

    optimizer = AdamW(
        [
            {"params": backbone_params,
             "lr": args.learning_rate * args.backbone_lr_scale,
             "name": "backbone"},
            {"params": phone_emb_params,
             "lr": args.learning_rate,
             "name": "phone_emb"},
        ],
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = max(len(train_loader) // args.grad_accumulation, 1)
    total_steps     = (args.max_steps if args.max_steps > 0
                       else args.epochs * steps_per_epoch)
    total_steps     = max(total_steps, 1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # ── AMP [C6] ──────────────────────────────────────────────────────────────
    # H100/H200: bfloat16 is preferred (wider dynamic range, no loss scaling needed)
    # bfloat16 is natively supported on Ampere+ (A100, H100, H200)
    use_amp   = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # GradScaler only needed for float16; bfloat16 does not need loss scaling
    scaler    = (torch.amp.GradScaler("cuda")
                 if use_amp and amp_dtype == torch.float16
                 else None)
    if use_amp:
        print(f"[AMP] Enabled: {amp_dtype}  |  GradScaler: {scaler is not None}")

    # ── EWC ───────────────────────────────────────────────────────────────────
    ewc: Optional[EWC] = None
    if args.ewc_lambda > 0:
        en_zh_specs = [s for s in train_specs if s.language in ("en", "zh")]
        if not en_zh_specs:
            print("[EWC] ⚠ No EN/ZH manifests — EWC disabled.")
        else:
            ewc_ds = T2SDataset(en_zh_specs, root_dir=args.root_dir)
            ewc_loader = DataLoader(
                ewc_ds,
                batch_size=4,           # small batch for Fisher: fine-grained gradient samples
                shuffle=True,
                num_workers=0,          # main process for determinism
                collate_fn=collate_t2s,
            )
            ewc = EWC(
                model, ewc_loader, device,
                lambda_=args.ewc_lambda,
                n_batches=args.ewc_fisher_batches,
            )

    # ── Resume [B3] ───────────────────────────────────────────────────────────
    global_step   = 0
    start_epoch   = 0
    best_val_loss = math.inf
    recent_ckpts: List[str] = []

    if args.resume:
        rpath = Path(args.resume).expanduser().resolve()
        if rpath.exists():
            print(f"[Resume] Loading: {rpath}")
            # [B3] weights_only=True — safe for pure state-dict checkpoints
            ckpt = torch.load(rpath, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scheduler") and scheduler:
                scheduler.load_state_dict(ckpt["scheduler"])
            if ckpt.get("scaler") and scaler:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch   = ckpt.get("epoch",         0)
            global_step   = ckpt.get("step",          0)
            best_val_loss = ckpt.get("best_val_loss",  math.inf)  # [F3]
            recent_ckpts  = ckpt.get("recent_checkpoints", [])
            print(
                f"[Resume] Epoch={start_epoch} | "
                f"Step={global_step} | "
                f"BestValLoss={best_val_loss:.4f}"   # [F3]
            )
        else:
            print(f"[Resume] ⚠ Checkpoint not found: {rpath} — starting fresh.")

    # ════════════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ════════════════════════════════════════════════════════════════════════════
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'═'*72}")
        print(
            f"  Epoch {epoch+1}/{args.epochs}  |  "
            f"GlobalStep={global_step}  |  "
            f"BestValLoss={best_val_loss:.4f}"
        )
        print(f"{'═'*72}")

        epoch_ce      = 0.0
        epoch_acc     = 0.0
        epoch_batches = 0
        # [C3] Dedicated counter — never incremented on None batches
        accum_counter = 0

        for batch in train_loader:
            if batch is None:      # [C3] skip without touching accum_counter
                continue

            codes      = batch["codes"].to(device, non_blocking=True)
            phones     = batch["phones"].to(device, non_blocking=True)
            x_mask     = batch["x_mask"].to(device, non_blocking=True)
            phone_mask = batch["phone_mask"].to(device, non_blocking=True)

            # ── Forward ──────────────────────────────────────────────────────
            # model.forward() path:
            #   phone_emb(phone_id) → (B, L, H)
            #   compute_loss → forward_diffusion:
            #     cfg_scale=0.15 → 85% real prompt, 15% null (CFG dropout)
            #     Bernoulli mask at sin(πt/2) rate, min 0.2
            #   DiffLlamaPrefix(xt, t, x_mask, phone_emb_prefix)
            #     → strips phone prefix → returns semantic hidden states
            #   to_logit() → (B, T, 8192)
            with torch.amp.autocast(
                "cuda", dtype=amp_dtype, enabled=use_amp
            ):
                logits, final_mask, x0, _, _ = model(
                    codes, x_mask, phone_id=phones, phone_mask=phone_mask
                )
                ce_loss  = _masked_ce_loss(logits, final_mask, x0)
                ewc_loss = (ewc.penalty(model)
                            if ewc is not None
                            else torch.tensor(0.0, device=device))
                loss     = (ce_loss + ewc_loss) / args.grad_accumulation

            # ── Backward ─────────────────────────────────────────────────────
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_counter += 1

            # ── Optimizer step ────────────────────────────────────────────────
            if accum_counter % args.grad_accumulation == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                acc = compute_accuracy(logits, final_mask, x0)
                epoch_ce      += ce_loss.item()
                epoch_acc     += acc
                epoch_batches += 1

                # ── Logging [C1] ──────────────────────────────────────────────
                if global_step % args.log_interval == 0:
                    lrs          = scheduler.get_last_lr()
                    lr_backbone  = lrs[0]
                    lr_phone_emb = lrs[1] if len(lrs) > 1 else lrs[0]
                    gn = (grad_norm.item()
                          if isinstance(grad_norm, torch.Tensor)
                          else float(grad_norm))

                    print(
                        f"  Step {global_step:6d} | "
                        f"CE={ce_loss.item():.4f} | "
                        f"Acc={acc:.3f} | "
                        f"LR_bb={lr_backbone:.2e} | "
                        f"LR_pe={lr_phone_emb:.2e} | "
                        f"GNorm={gn:.3f}"
                        + (f" | EWC={ewc_loss.item():.4f}" if ewc else "")
                    )

                    writer.add_scalar("Train/loss_ce",   ce_loss.item(), global_step)
                    writer.add_scalar("Train/top1_acc",  acc,            global_step)
                    writer.add_scalar("Train/grad_norm", gn,             global_step)
                    writer.add_scalar("LR/backbone",     lr_backbone,    global_step)
                    writer.add_scalar("LR/phone_emb",    lr_phone_emb,   global_step)
                    if ewc:
                        writer.add_scalar("Train/loss_ewc",
                                          ewc_loss.item(), global_step)
                        writer.add_scalar("Train/loss_total",
                                          ce_loss.item() + ewc_loss.item(),
                                          global_step)

                    # Per-language CE from current batch
                    languages = batch["languages"]
                    for lang in set(languages):
                        idx_list = [i for i, l in enumerate(languages) if l == lang]
                        if not idx_list:
                            continue
                        idx_t   = torch.tensor(idx_list, device=device)
                        lang_ce = _masked_ce_loss(
                            logits[idx_t], final_mask[idx_t], x0[idx_t]
                        ).item()
                        writer.add_scalar(
                            f"Train/loss_{lang}", lang_ce, global_step
                        )

                # ── Validation ────────────────────────────────────────────────
                if args.val_interval > 0 and global_step % args.val_interval == 0:
                    val_metrics = evaluate(model, val_loader, device)
                    val_str = " | ".join(
                        f"{k}={v:.4f}" for k, v in val_metrics.items()
                    )
                    print(f"  [Val] Step {global_step}: {val_str}")
                    for k, v in val_metrics.items():
                        writer.add_scalar(f"Val/{k}", v, global_step)

                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        best_path     = output_dir / "best_model.pth"
                        save_checkpoint(
                            best_path, model, optimizer, scheduler, scaler,
                            epoch, global_step, best_val_loss, recent_ckpts
                        )
                        print(
                            f"  [Val] ✓ New best: "
                            f"loss={best_val_loss:.4f} → {best_path.name}"
                        )

                # ── Step checkpoint [C4]  [F3]  [F4] ─────────────────────────
                if global_step % args.save_every == 0:
                    ckpt_path = output_dir / f"step_{global_step:07d}.pth"
                    # [C4] Append before save so saved state has correct list
                    recent_ckpts.append(str(ckpt_path))
                    recent_ckpts = prune_old_checkpoints(recent_ckpts, args.keep_last)
                    save_checkpoint(
                        ckpt_path, model, optimizer, scheduler, scaler,
                        epoch, global_step, best_val_loss, recent_ckpts  # [F3]
                    )
                    # [F4] No latest.pth — resume with explicit --resume path
                    print(f"  [Ckpt] Saved {ckpt_path.name}")

                if args.max_steps > 0 and global_step >= args.max_steps:
                    print(f"[Main] max_steps={args.max_steps} reached.")
                    break

        # ── End-of-epoch flush [B4] ───────────────────────────────────────────
        # If len(train_loader) % grad_accumulation != 0, partial accumulation
        # window gradients remain in .grad tensors. Flush them so they do not
        # contaminate the first optimizer step of the next epoch.
        optimizer.zero_grad(set_to_none=True)

        if epoch_batches > 0:
            avg_ce  = epoch_ce  / epoch_batches
            avg_acc = epoch_acc / epoch_batches
            writer.add_scalar("Epoch/train_loss", avg_ce,  epoch)
            writer.add_scalar("Epoch/train_acc",  avg_acc, epoch)
            print(
                f"  [Epoch {epoch+1}] "
                f"AvgCE={avg_ce:.4f} | "
                f"AvgAcc={avg_acc:.3f} | "
                f"Steps={epoch_batches}"
            )

        # val_interval=0 → validate once per epoch  [C5]
        if args.val_interval == 0:
            val_metrics = evaluate(model, val_loader, device)
            val_str = " | ".join(
                f"{k}={v:.4f}" for k, v in val_metrics.items()
            )
            print(f"  [Val/Epoch {epoch+1}] {val_str}")
            for k, v in val_metrics.items():
                writer.add_scalar(f"Epoch/val_{k}", v, epoch)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_path     = output_dir / "best_model.pth"
                save_checkpoint(
                    best_path, model, optimizer, scheduler, scaler,
                    epoch, global_step, best_val_loss, recent_ckpts
                )
                print(
                    f"  [Val] ✓ New best: "
                    f"loss={best_val_loss:.4f} → {best_path.name}"
                )

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = output_dir / f"final_epoch{args.epochs}_step{global_step}.pth"
    recent_ckpts.append(str(final_path))
    recent_ckpts = prune_old_checkpoints(recent_ckpts, args.keep_last)
    save_checkpoint(
        final_path, model, optimizer, scheduler, scaler,
        args.epochs, global_step, best_val_loss, recent_ckpts
    )
    print(f"\n[Main] Training complete.")
    print(f"  Final checkpoint : {final_path}")
    print(f"  Best val loss    : {best_val_loss:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
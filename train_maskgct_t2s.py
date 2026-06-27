#!/usr/bin/env python3
"""
train_maskgct_t2s.py — v13
Continuous-learning T2S trainer: Tamil + Telugu + Malayalam + Kannada + Hindi + EN + ZH.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIXES (v12 → v13)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[v13-FIX-A] _accum_epoch_lang_from_raw: accumulate raw token sums
  instead of sample-weighted averages of batch-level token averages.

  Previous (v12):
    epoch_lang[lang] = [sum(token_avg_CE × n_samples),
                        sum(token_avg_acc × n_samples),
                        n_samples_total]
    Final = epoch_lang[lang][0] / epoch_lang[lang][2]
          = weighted avg of BATCH-LEVEL token averages, weighted by
            sample count — biased when sequences have different lengths.

  Correct (v13):
    epoch_lang[lang] = [sum_raw_token_CE,
                        sum_correct_tokens,
                        sum_valid_tokens]
    Final = epoch_lang[lang][0] / epoch_lang[lang][2]
          = true token-level average over all tokens seen this epoch.

  Concrete example (from bug report):
    Batch 1: 5 short samples, 50 total tokens, token-avg CE = 1.0
    Batch 2: 3 long  samples, 300 total tokens, token-avg CE = 2.0
    v12 result: (1.0×5 + 2.0×3) / 8     = 1.375  ← biased
    v13 result: (1.0×50 + 2.0×300) / 350 = 1.857  ← correct

[v13-FIX-B] evaluate(): derive overall_loss / overall_acc from the
  same raw token sums in lang_agg (sum_ce, sum_correct, sum_valid)
  rather than accumulating sample-weighted loss_scalar × bsz.
  The total_loss / total_acc / count variables are removed entirely;
  overall metrics are computed as:
    overall_loss = Σ lang_agg[l][0] / Σ lang_agg[l][2]
    overall_acc  = Σ lang_agg[l][1] / Σ lang_agg[l][2]

[v13-FIX-C] main() epoch summary: removed epoch_ce and epoch_acc
  accumulators. Overall epoch metrics now derive from epoch_lang raw
  sums (same token-weighted formula as v13-FIX-A/B).

Retained from v12: v12-FIX-A (detach), v12-FIX-B (1 CE pass in
evaluate), v11 (log_interval gating), v10-FIX-A/B (table borders),
and all prior fixes FIX #1–#7, F1–F11.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import datetime
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup
from huggingface_hub import hf_hub_download
import safetensors.torch

from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
from utils.util import load_config

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

ALL_LANGUAGES: List[str] = ["ta", "te", "ml", "kn", "hi", "en", "zh"]

INDIC_PHONE_START = 363
INDIC_PHONE_END   = 392   # inclusive — re-initialized at N(0, 0.02)

PHONE_PAD_ID = 1023
MAX_PHONE_ID = 1022
MAX_CODE_ID  = 8191

PHONE_EMB_VOCAB_SIZE = 1024

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "MaskGCT T2S continual-learning finetuning "
            "(Tamil + Telugu + Malayalam + Kannada + Hindi + EN + ZH)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--root-dir",   type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument(
        "--train-manifest", dest="train_manifests", action="append", required=True,
        metavar="PATH::lang",
    )
    p.add_argument(
        "--val-manifest", dest="val_manifests", action="append", required=True,
        metavar="PATH::lang",
    )

    p.add_argument(
        "--config", type=Path,
        default=Path("models/tts/maskgct/config/maskgct.json"),
    )
    p.add_argument("--base-checkpoint", type=Path, required=True)

    p.add_argument("--ta-phone-offset",    type=int, default=363)
    p.add_argument("--indic-phone-offset", type=int, default=380)

    p.add_argument("--batch-size",        type=int,   default=32)
    p.add_argument("--grad-accumulation", type=int,   default=4)
    p.add_argument("--epochs",            type=int,   default=30)
    p.add_argument("--max-steps",         type=int,   default=0)
    p.add_argument("--learning-rate",     type=float, default=2e-5)
    p.add_argument("--backbone-lr-scale", type=float, default=0.1)
    p.add_argument("--weight-decay",      type=float, default=0.01)
    p.add_argument("--warmup-steps",      type=int,   default=500)
    p.add_argument("--grad-clip",         type=float, default=1.0)
    p.add_argument("--amp",               action="store_true")
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--seed",              type=int,   default=42)

    p.add_argument(
        "--lang-balance", type=str,
        default="ta:4,te:2,ml:2,kn:2,hi:2,en:2,zh:2",
    )

    p.add_argument("--ewc-lambda",         type=float, default=0.0)
    p.add_argument("--ewc-fisher-batches", type=int,   default=200)

    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--val-interval", type=int, default=500)
    p.add_argument("--save-every",   type=int, default=1000)
    p.add_argument("--keep-last",    type=int, default=3)
    p.add_argument("--resume",       type=str, default="")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Manifest parsing
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ManifestSpec:
    path: Path
    language: Optional[str] = None


def parse_manifest_specs(entries: Sequence[str], flag_name: str) -> List[ManifestSpec]:
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
    id: str
    semantic_codes_path: Path
    phone_ids_path: Path
    language: str
    code_len: int
    phone_len: int


class T2SDataset(Dataset):
    def __init__(self, specs: Sequence[ManifestSpec], root_dir: Path) -> None:
        self.root_dir    = root_dir.resolve()
        self.samples:         List[T2SSample]      = []
        self.bad_indices:     Set[int]             = set()
        self.lang_to_indices: Dict[str, List[int]] = defaultdict(list)

        for spec in specs:
            self._load_manifest(spec)

        unknown = self.lang_to_indices.get("unknown", [])
        if unknown:
            print(
                f"\n[Dataset] ⚠ {len(unknown)} samples have language='unknown'. "
                f"Add '::lang' suffix to manifest flags.\n"
            )

        if not self.samples:
            raise RuntimeError("No samples loaded.")

        print(f"\n[Dataset] {len(self.samples):,} total samples:")
        for lang in ALL_LANGUAGES + ["unknown"]:
            idxs = self.lang_to_indices.get(lang, [])
            if idxs:
                print(f"  [{lang:>4}] {len(idxs):>8,}")

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
                    print(f"  [Skip] line {lineno}: {e}")
                    skipped += 1
                    continue

                codes_rel  = rec.get("semantic_codes_path") or rec.get("codes_path", "")
                phones_rel = rec.get("phone_ids_path") or rec.get("text_ids_path", "")
                lang = (rec.get("language") or spec.language or "unknown").lower()

                if not codes_rel or not phones_rel:
                    skipped += 1
                    continue

                idx = len(self.samples)
                self.samples.append(T2SSample(
                    id=rec.get("id", f"sample_{idx}"),
                    semantic_codes_path=self.root_dir / lang / lang / codes_rel,
                    phone_ids_path=self.root_dir / lang / lang / phones_rel,
                    language=lang,
                    code_len=int(rec.get("code_len", 0)),
                    phone_len=int(rec.get("phone_len", 0)),
                ))
                self.lang_to_indices[lang].append(idx)
                loaded += 1

        print(f"  {spec.path.name} (lang={spec.language or 'from_field'}): "
              f"loaded={loaded:,}, skipped={skipped}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        if idx in self.bad_indices:
            return None
        sample = self.samples[idx]
        try:
            codes  = np.load(sample.semantic_codes_path, allow_pickle=False).astype(np.int64)
            phones = np.load(sample.phone_ids_path,      allow_pickle=False).astype(np.int64)

            if codes.size == 0 or phones.size == 0:
                raise ValueError("Empty .npy")
            if codes.max() > MAX_CODE_ID:
                raise ValueError(f"code {codes.max()} > {MAX_CODE_ID}")
            if phones.min() < 0:
                raise ValueError(f"Negative phone ID {phones.min()} — corrupted .npy")
            if phones.max() > MAX_PHONE_ID:
                raise ValueError(f"phone {phones.max()} > {MAX_PHONE_ID}")

            code_len  = int(codes.shape[0])
            phone_len = int(phones.shape[0])
            if sample.code_len  == 0: sample.code_len  = code_len
            if sample.phone_len == 0: sample.phone_len = phone_len

            return {
                "id":        sample.id,
                "codes":     torch.from_numpy(codes),
                "phones":    torch.from_numpy(phones),
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
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    codes_padded  = pad_sequence([b["codes"]  for b in batch],
                                  batch_first=True, padding_value=0)
    phones_padded = pad_sequence([b["phones"] for b in batch],
                                  batch_first=True, padding_value=PHONE_PAD_ID)

    B = len(batch)
    x_mask     = torch.zeros(B, codes_padded.shape[1],  dtype=torch.float32)
    phone_mask = torch.zeros(B, phones_padded.shape[1], dtype=torch.float32)
    for i, b in enumerate(batch):
        x_mask[i,     : b["code_len"]]  = 1.0
        phone_mask[i, : b["phone_len"]] = 1.0

    return {
        "codes":      codes_padded,
        "phones":     phones_padded,
        "x_mask":     x_mask,
        "phone_mask": phone_mask,
        "languages":  [b["language"] for b in batch],
        "ids":        [b["id"]       for b in batch],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Sampler
# ══════════════════════════════════════════════════════════════════════════════

def parse_lang_balance(spec: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lang, _, wt = part.partition(":")
        result[lang.strip()] = float(wt.strip()) if wt.strip() else 1.0
    return result


def build_weighted_sampler(
    dataset: T2SDataset, lang_weights: Dict[str, float]
) -> WeightedRandomSampler:
    per_lang_w: Dict[str, float] = {}
    for lang, weight in lang_weights.items():
        count = len(dataset.lang_to_indices.get(lang, []))
        if count == 0:
            print(f"[Sampler] ⚠ '{lang}' has 0 samples — skipped.")
            continue
        per_lang_w[lang] = weight / count

    sample_weights = [per_lang_w.get(s.language, 1e-9) for s in dataset.samples]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# Phone embedding lookup
# ══════════════════════════════════════════════════════════════════════════════

def _find_phone_embedding(model: MaskGCT_T2S) -> Tuple[str, nn.Embedding]:
    """
    Search order:
    1. Known attribute names (fast path)
    2. Walk: exact vocab size + padding_idx  (safest fallback)
    3. Walk: exact vocab size only           (last resort, prints warning)
    """
    for candidate in ("phone_embedding", "phone_emb", "phone_embed"):
        mod = getattr(model, candidate, None)
        if isinstance(mod, nn.Embedding):
            return candidate, mod

    for name, mod in model.named_modules():
        if (isinstance(mod, nn.Embedding)
                and mod.num_embeddings == PHONE_EMB_VOCAB_SIZE
                and mod.padding_idx    == PHONE_PAD_ID):
            print(f"[Model] ⚠ Phone embedding found via walk: '{name}'")
            return name, mod

    for name, mod in model.named_modules():
        if (isinstance(mod, nn.Embedding)
                and mod.num_embeddings == PHONE_EMB_VOCAB_SIZE):
            print(f"[Model] ⚠ Phone embedding found via walk (no padding_idx match): '{name}'")
            return name, mod

    raise RuntimeError(
        "Cannot find phone embedding in MaskGCT_T2S. "
        f"Expected nn.Embedding with num_embeddings=={PHONE_EMB_VOCAB_SIZE} "
        f"and padding_idx=={PHONE_PAD_ID}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Config loader
# ══════════════════════════════════════════════════════════════════════════════

def _get_nested(obj, path: str):
    for key in path.split("."):
        if not key:
            continue
        if isinstance(obj, dict):
            obj = obj[key]
        else:
            obj = getattr(obj, key)
    return obj


def _load_t2s_config(cfg_path: Path):
    cfg = load_config(str(cfg_path))
    for path in ("model.t2s_model", "t2s_model", ""):
        try:
            obj = _get_nested(cfg, path)
            print(f"[Config] Using path: '{path or '<root>'}'")
            return obj
        except (AttributeError, KeyError):
            continue
    raise RuntimeError(
        f"Cannot resolve t2s sub-config from {cfg_path}. "
        "Tried: model.t2s_model, t2s_model, root."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Model builder
# ══════════════════════════════════════════════════════════════════════════════

def build_t2s_model(
    cfg_path:        Path,
    device:          torch.device,
) -> Tuple[MaskGCT_T2S, str, nn.Embedding]:
    t2s_cfg = _load_t2s_config(cfg_path)
    model   = MaskGCT_T2S(cfg=t2s_cfg)
    # download t2s model ckpt
    t2s_model_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="t2s_model/model.safetensors"
    )
    print(f"[Model] Loading: {t2s_model_ckpt}")
    missing, unexpected = safetensors.torch.load_model(
        model, t2s_model_ckpt, strict=False
    )
    if missing:
        print(f"  Missing    ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected ({len(unexpected)}): {unexpected[:5]}")

    phone_attr, phone_emb_mod = _find_phone_embedding(model)
    print(f"[Model] Phone embedding: '{phone_attr}' "
          f"({phone_emb_mod.num_embeddings} × {phone_emb_mod.embedding_dim}d, "
          f"padding_idx={phone_emb_mod.padding_idx})")

    with torch.no_grad():
        nn.init.normal_(
            phone_emb_mod.weight[INDIC_PHONE_START : INDIC_PHONE_END + 1],
            mean=0.0, std=0.02,
        )
    print(
        f"[Model] Re-initialized rows {INDIC_PHONE_START}–{INDIC_PHONE_END} "
        f"N(0, 0.02)  (Tamil: 363–379 | te/ml/kn/hi: 380–392)"
    )

    model = model.to(device)

    frozen    = [(n, p.shape) for n, p in model.named_parameters() if not p.requires_grad]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Model] {total/1e6:.1f}M total | {trainable/1e6:.1f}M trainable")
    if frozen:
        print(f"[Model] ⚠ {len(frozen)} frozen tensors:")
        for n, s in frozen[:10]:
            print(f"    {n}  {list(s)}")

    return model, phone_attr, phone_emb_mod


# ══════════════════════════════════════════════════════════════════════════════
# Gradient hook — protect pretrained rows 0–362
# ══════════════════════════════════════════════════════════════════════════════

def register_phone_emb_hook(
    phone_emb_mod: nn.Embedding,
    protect_up_to: int = 362,
) -> torch.utils.hooks.RemovableHook:
    """
    Zeros gradients for rows 0..protect_up_to after every backward.
    Only rows 363–392 (new Indic phonemes) accumulate gradients.

    This hook also fires during EWC Fisher estimation, so Fisher for
    rows 0–362 is zero — EWC won't protect them. This is correct: the
    hook already prevents their update (a stronger guarantee than EWC).
    """
    def _zero_pretrained_rows(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[: protect_up_to + 1] = 0.0
        return grad

    return phone_emb_mod.weight.register_hook(_zero_pretrained_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Forward wrapper
# ══════════════════════════════════════════════════════════════════════════════

_fwd_state: Dict[str, Optional[str]] = {"kwarg": None}


def _model_forward(
    model:      MaskGCT_T2S,
    codes:      torch.Tensor,
    x_mask:     torch.Tensor,
    phones:     torch.Tensor,
    phone_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cached = _fwd_state["kwarg"]

    if cached is None:
        for kwarg in ("phone_ids", "phone_id"):
            try:
                raw = model(codes, x_mask, **{kwarg: phones, "phone_mask": phone_mask})
                _fwd_state["kwarg"] = kwarg
                break
            except TypeError:
                continue
        else:
            try:
                raw = model(codes, x_mask, phones, phone_mask)
                _fwd_state["kwarg"] = ""
            except TypeError as e:
                raise RuntimeError(
                    f"Cannot call model.forward with any known signature: {e}"
                ) from e
    elif cached == "":
        raw = model(codes, x_mask, phones, phone_mask)
    else:
        raw = model(codes, x_mask, **{cached: phones, "phone_mask": phone_mask})

    logits     = raw[0]
    final_mask = raw[1]
    x0         = raw[2] if len(raw) > 2 else codes.clone()

    if final_mask.dim() == 2:
        final_mask = final_mask.unsqueeze(-1)

    return logits, final_mask, x0


# ══════════════════════════════════════════════════════════════════════════════
# EWC
# ══════════════════════════════════════════════════════════════════════════════

class EWC:
    """L_total = L_ce + (λ/2) * Σ_i F_i * (θ_i − θ*_i)²"""

    def __init__(
        self,
        model:            MaskGCT_T2S,
        reference_loader: DataLoader,
        device:           torch.device,
        lambda_:          float       = 5000.0,
        n_batches:        int         = 200,
        use_amp:          bool        = False,
        amp_dtype:        torch.dtype = torch.float32,
    ) -> None:
        self.lambda_ = lambda_
        self.device  = device

        self._params: Dict[str, torch.Tensor] = {
            name: param.clone().detach().to(device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        print(f"[EWC] Estimating Fisher ({n_batches} EN/ZH batches, amp={use_amp})...")
        self._fisher = self._compute_fisher(
            model, reference_loader, n_batches, use_amp, amp_dtype
        )
        print(f"[EWC] Ready. λ={lambda_:.0f} | params tracked: {len(self._fisher)}")

    def _compute_fisher(
        self,
        model:     MaskGCT_T2S,
        loader:    DataLoader,
        n_batches: int,
        use_amp:   bool,
        amp_dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
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
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits, final_mask, x0 = _model_forward(
                    model, codes, x_mask, phones, phone_mask
                )
                loss = _masked_ce_loss(logits, final_mask, x0)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad = param.grad
                    if grad.is_sparse:
                        grad = grad.to_dense()
                    fisher[name] += grad.data.pow(2)
            batch_count += 1

        for name in fisher:
            fisher[name] /= max(batch_count, 1)

        model.zero_grad()
        print(f"[EWC] Fisher estimated over {batch_count} batches.")
        return fisher

    def penalty(self, model: MaskGCT_T2S) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self._fisher:
                continue
            loss = loss + (
                self._fisher[name] * (param - self._params[name]).pow(2)
            ).sum()
        return (self.lambda_ / 2.0) * loss


# ══════════════════════════════════════════════════════════════════════════════
# Loss helpers
# ══════════════════════════════════════════════════════════════════════════════

def _masked_ce_loss(
    logits:     torch.Tensor,   # (B, T, V)
    final_mask: torch.Tensor,   # (B, T, 1)
    x0:         torch.Tensor,   # (B, T)
) -> torch.Tensor:
    B, T, V = logits.shape
    mask_f  = final_mask.reshape(B * T)
    ce_all  = F.cross_entropy(
        logits.reshape(B * T, V), x0.reshape(B * T), reduction="none"
    )
    return (ce_all * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def _masked_ce_loss_with_raw(
    logits:     torch.Tensor,   # (B, T, V)
    final_mask: torch.Tensor,   # (B, T, 1)
    x0:         torch.Tensor,   # (B, T)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Same as _masked_ce_loss but also returns raw (ce_all, mask_f) for
    downstream per-lang accumulation without re-running F.cross_entropy.

    [v12-FIX-A] ce_all and mask_f are detached before returning so the
    autograd graph (including saved logits B×T×V) is freed immediately
    after loss.backward() — not held across gradient accumulation steps.

    Returns: (loss_scalar, ce_all_BT_detached, mask_f_BT_detached)
    """
    B, T, V = logits.shape
    mask_f  = final_mask.reshape(B * T)
    ce_all  = F.cross_entropy(
        logits.reshape(B * T, V), x0.reshape(B * T), reduction="none"
    )
    loss = (ce_all * mask_f).sum() / mask_f.sum().clamp(min=1.0)
    return loss, ce_all.detach(), mask_f.detach()   # [v12-FIX-A]


@torch.no_grad()
def compute_accuracy(
    logits:     torch.Tensor,
    final_mask: torch.Tensor,
    x0:         torch.Tensor,
) -> float:
    preds = logits.argmax(dim=-1)
    mask  = final_mask.squeeze(-1)
    valid = mask.sum().item()
    return ((preds == x0).float() * mask).sum().item() / valid if valid > 0 else 0.0


def per_lang_metrics(
    logits:     torch.Tensor,
    final_mask: torch.Tensor,
    x0:         torch.Tensor,
    languages:  List[str],
    device:     torch.device,
) -> Dict[str, Dict[str, float]]:
    """Called only at log_interval steps (training) for live console logging."""
    results: Dict[str, Dict[str, float]] = {}
    for lang in set(languages):
        idx_t = torch.tensor(
            [i for i, l in enumerate(languages) if l == lang], device=device
        )
        results[lang] = {
            "loss": _masked_ce_loss(logits[idx_t], final_mask[idx_t], x0[idx_t]).item(),
            "acc":  compute_accuracy(logits[idx_t], final_mask[idx_t], x0[idx_t]),
        }
    return results


@torch.no_grad()
def _accum_epoch_lang_from_raw(
    ce_all:     torch.Tensor,   # (B*T,) raw per-token CE — already detached
    mask_f:     torch.Tensor,   # (B*T,) mask — already detached
    logits:     torch.Tensor,   # (B, T, V) for accuracy
    final_mask: torch.Tensor,   # (B, T, 1)
    x0:         torch.Tensor,   # (B, T)
    languages:  List[str],
    device:     torch.device,
    epoch_lang: Dict,           # mutated in-place
) -> None:
    """
    Accumulates raw token-level sums for mathematically correct averages.

    [v13-FIX-A] epoch_lang[lang] stores:
      [0] sum_raw_token_CE    — Σ ce(token) for valid tokens of this lang
      [1] sum_correct_tokens  — Σ 1(pred==target) for valid tokens
      [2] sum_valid_tokens    — Σ mask value (total valid token count)

    Final metrics = [0]/[2] and [1]/[2] → true token-weighted averages
    regardless of sequence length variation across batches or languages.

    Previous versions (v11/v12) stored token_avg × n_samples in [0]/[1]
    and n_samples in [2], producing biased sample-weighted averages when
    sequence lengths were unequal.
    """
    B, T = x0.shape
    ce_BT   = ce_all.reshape(B, T)
    mask_BT = mask_f.reshape(B, T)
    preds   = logits.argmax(dim=-1)     # (B, T)
    mask2d  = final_mask.squeeze(-1)    # (B, T)

    for lang in set(languages):
        rows  = [i for i, l in enumerate(languages) if l == lang]
        r     = torch.tensor(rows, device=device)
        m     = mask_BT[r]                                              # (n, T)
        epoch_lang[lang][0] += (ce_BT[r] * m).sum().item()             # sum_raw_ce
        epoch_lang[lang][1] += ((preds[r] == x0[r]).float()
                                 * mask2d[r]).sum().item()              # sum_correct
        epoch_lang[lang][2] += m.sum().item()                          # sum_valid_tokens


# ══════════════════════════════════════════════════════════════════════════════
# Phone-offset verification
# ══════════════════════════════════════════════════════════════════════════════

def verify_phone_offsets(
    dataset:            T2SDataset,
    ta_phone_offset:    int,
    indic_phone_offset: int,
    n_check:            int = 50,
) -> None:
    checks = [
        ("ta",          ta_phone_offset,    ["ta"]),
        ("te/ml/kn/hi", indic_phone_offset, ["te", "ml", "kn", "hi"]),
    ]
    for label, expected, lang_list in checks:
        indices: List[int] = []
        for lang in lang_list:
            indices.extend(dataset.lang_to_indices.get(lang, []))
        if not indices:
            print(f"[Verify] No {label} samples — skipping.")
            continue
        sample_idxs = random.sample(indices, min(n_check, len(indices)))
        max_id  = 0
        checked = 0
        for idx in sample_idxs:
            try:
                phones = np.load(dataset.samples[idx].phone_ids_path, allow_pickle=False)
                if phones.size > 0:
                    max_id = max(max_id, int(phones.max()))
                checked += 1
            except Exception as e:
                print(f"  [Verify] read error: {e}")
        if checked == 0:
            print(f"[Verify] ⚠ All {label} spot-check reads failed.")
            continue
        if max_id < expected:
            print(
                f"\n[Verify] ⚠ {label}: expected max phone ID >= {expected}, "
                f"got {max_id} over {checked} samples.\n"
            )
        else:
            print(f"[Verify] ✓ {label}: max_id={max_id} >= {expected} ({checked} samples)")


# ══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    path:               Path,
    model:              nn.Module,
    optimizer:          torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch:              int,
    step:               int,
    best_val_loss:      float,
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
            "best_val_loss":      best_val_loss,
            "recent_checkpoints": recent_checkpoints,
        },
        path,
    )


def prune_old_checkpoints(recent: List[str], keep_last: int) -> List[str]:
    while len(recent) > keep_last:
        old = recent.pop(0)
        try:
            if os.path.exists(old):
                os.remove(old)
                print(f"[Ckpt] Pruned: {Path(old).name}")
        except OSError as e:
            print(f"[Ckpt] Remove failed: {e}")
    return recent


# ══════════════════════════════════════════════════════════════════════════════
# Validation  [v13-FIX-B] — token-weighted overall metrics from lang_agg sums
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model:     MaskGCT_T2S,
    loader:    DataLoader,
    device:    torch.device,
    step:      int,
    use_amp:   bool        = False,
    amp_dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """
    Validation loop.  All metrics are true token-weighted averages.

    [v13-FIX-B] overall_loss / overall_acc are now derived from the raw
    token sums in lang_agg rather than accumulating loss_scalar × bsz.
    The removed total_loss / total_acc / count variables were biased
    when sequence lengths varied across batches (sample-weighted instead
    of token-weighted).

    [v12-FIX-B] Single F.cross_entropy call per batch via
    _masked_ce_loss_with_raw + _accum_epoch_lang_from_raw.

    Table uses correct widths: PER_LANG_WIDTH=22, top border computed
    from table_inner_width (v10-FIX-A/B).
    """
    model.eval()

    # lang_agg[lang] = [sum_raw_token_CE, sum_correct_tokens, sum_valid_tokens]
    lang_agg: Dict[str, List] = defaultdict(lambda: [0.0, 0.0, 0.0])

    for batch in loader:
        if batch is None:
            continue
        codes      = batch["codes"].to(device)
        phones     = batch["phones"].to(device)
        x_mask     = batch["x_mask"].to(device)
        phone_mask = batch["phone_mask"].to(device)
        languages  = batch["languages"]

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits, final_mask, x0 = _model_forward(
                model, codes, x_mask, phones, phone_mask
            )
            # [v12-FIX-B] Single CE pass; ce_all/mask_f reused for per-lang agg
            _, ce_all, mask_f = _masked_ce_loss_with_raw(logits, final_mask, x0)

        # Accumulate raw token sums — token-weighted [v13-FIX-B]
        _accum_epoch_lang_from_raw(
            ce_all, mask_f, logits, final_mask, x0,
            languages, device, lang_agg
        )

    model.train()

    # Derive overall metrics from raw token sums — unbiased [v13-FIX-B]
    total_sum_ce      = sum(v[0] for v in lang_agg.values())
    total_sum_correct = sum(v[1] for v in lang_agg.values())
    total_sum_m       = sum(v[2] for v in lang_agg.values())

    if total_sum_m == 0.0:
        print(f"  [Val] Step {step}: no samples.")
        return {"loss": 0.0, "top1_acc": 0.0}

    overall_loss = total_sum_ce      / total_sum_m
    overall_acc  = total_sum_correct / total_sum_m

    # ── Validation table (v10-FIX-A/B) ──────────────────────────────────────
    present = [l for l in ALL_LANGUAGES if l in lang_agg]
    cw = 10

    # Each lang column pair: f" {field:>{cw}} {field:>{cw}}" = (1+cw)+(1+cw) = 22 chars
    PER_LANG_WIDTH    = 2 + 2 * cw   # 22
    table_inner_width = 6 + PER_LANG_WIDTH * len(present)

    header_inner = f"{'Lang':<6}"
    for lang in present:
        header_inner += f" {lang + ' loss':>{cw}} {lang + ' acc':>{cw}}"

    data_inner = f"{'per-lang':<6}"
    for lang in present:
        sm = max(lang_agg[lang][2], 1e-9)
        l_val = lang_agg[lang][0] / sm
        a_val = lang_agg[lang][1] / sm
        data_inner += f" {l_val:>{cw}.4f} {a_val:>{cw}.4f}"

    overall_label = "OVERALL"
    overall_str   = f"loss={overall_loss:.4f}  acc={overall_acc:.4f}"
    overall_pad   = table_inner_width - len(overall_label)
    overall_inner = f"{overall_label}" + f" {overall_str:<{overall_pad - 1}}"

    sep_inner = "─" * table_inner_width

    # v10-FIX-A: top border interior = table_inner_width + 2 chars total
    top_label  = f" Val @ step {step} "
    top_fill   = "─" * max(0, table_inner_width - len(top_label))
    top_border = f"┌─{top_label}{top_fill}─┐"

    print()
    print(f" {top_border}")
    print(f" │ {header_inner:<{table_inner_width}} │")
    print(f" ├─{sep_inner}─┤")
    print(f" │ {data_inner:<{table_inner_width}} │")
    print(f" ├─{sep_inner}─┤")
    print(f" │ {overall_inner:<{table_inner_width}} │")
    print(f" └─{sep_inner}─┘")
    print()

    results: Dict[str, float] = {"loss": overall_loss, "top1_acc": overall_acc}
    for lang, agg in lang_agg.items():
        sm = max(agg[2], 1e-9)
        results[f"loss_{lang}"] = agg[0] / sm
        results[f"acc_{lang}"]  = agg[1] / sm
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device  : {device} "
          f"({'cuda: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    print(f"[Main] PyTorch : {torch.__version__}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = os.environ.get(
        "T2S_RUN_NAME",
        f"t2s_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    writer = SummaryWriter(log_dir=str(output_dir / "logs" / run_name))

    model, phone_attr, phone_emb_mod = build_t2s_model(
        args.config, device
    )

    train_specs   = parse_manifest_specs(args.train_manifests, "--train-manifest")
    val_specs     = parse_manifest_specs(args.val_manifests,   "--val-manifest")
    train_dataset = T2SDataset(train_specs, root_dir=args.root_dir)
    val_dataset   = T2SDataset(val_specs,   root_dir=args.root_dir)

    verify_phone_offsets(train_dataset, args.ta_phone_offset, args.indic_phone_offset)

    lang_weights  = parse_lang_balance(args.lang_balance)
    train_sampler = build_weighted_sampler(train_dataset, lang_weights)

    pin = device.type == "cuda"
    pw  = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=collate_t2s,
        pin_memory=pin, persistent_workers=pw, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_t2s,
        pin_memory=pin, persistent_workers=pw,
    )

    backbone_params  = [p for n, p in model.named_parameters()
                        if p.requires_grad and phone_attr not in n]
    phone_emb_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and phone_attr in n]

    print(f"[Optim] backbone : {sum(p.numel() for p in backbone_params)/1e6:.1f}M "
          f"lr={args.learning_rate * args.backbone_lr_scale:.2e}")
    print(f"[Optim] phone_emb: {sum(p.numel() for p in phone_emb_params)/1e6:.3f}M "
          f"lr={args.learning_rate:.2e}  (rows 0–362 grad-zeroed via hook)")

    optimizer = AdamW(
        [
            {"params": backbone_params,  "lr": args.learning_rate * args.backbone_lr_scale},
            {"params": phone_emb_params, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )

    hook = register_phone_emb_hook(phone_emb_mod, protect_up_to=362)

    steps_per_epoch = max(math.ceil(len(train_loader) / args.grad_accumulation), 1)
    total_steps     = (args.max_steps if args.max_steps > 0
                       else args.epochs * steps_per_epoch)
    total_steps     = max(total_steps, 1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp   = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler    = (torch.amp.GradScaler("cuda")
                 if use_amp and amp_dtype == torch.float16 else None)
    if use_amp:
        print(f"[AMP] {amp_dtype} | GradScaler: {scaler is not None}")

    ewc: Optional[EWC] = None
    if args.ewc_lambda > 0:
        en_zh_specs = [s for s in train_specs if s.language in ("en", "zh")]
        if not en_zh_specs:
            print("[EWC] ⚠ No EN/ZH manifests — EWC disabled.")
        else:
            ewc_ds     = T2SDataset(en_zh_specs, root_dir=args.root_dir)
            ewc_loader = DataLoader(ewc_ds, batch_size=4, shuffle=True,
                                    num_workers=0, collate_fn=collate_t2s)
            ewc = EWC(
                model, ewc_loader, device,
                lambda_=args.ewc_lambda,
                n_batches=args.ewc_fisher_batches,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

    global_step   = 0
    start_epoch   = 0
    best_val_loss = math.inf
    recent_ckpts: List[str] = []

    if args.resume:
        rpath = Path(args.resume).expanduser().resolve()
        if rpath.exists():
            print(f"[Resume] Loading: {rpath}")
            ckpt = torch.load(rpath, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scheduler") and scheduler:
                scheduler.load_state_dict(ckpt["scheduler"])
            if ckpt.get("scaler") and scaler:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch   = ckpt.get("epoch",              0)
            global_step   = ckpt.get("step",               0)
            best_val_loss = ckpt.get("best_val_loss",  math.inf)
            recent_ckpts  = ckpt.get("recent_checkpoints", [])
            del ckpt
            print(f"[Resume] Epoch={start_epoch} | Step={global_step} | "
                  f"BestValLoss={best_val_loss:.4f}")
        else:
            print(f"[Resume] ⚠ Not found: {rpath} — starting fresh.")

    # ════════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ════════════════════════════════════════════════════════════════════════
    model.train()
    optimizer.zero_grad(set_to_none=True)

    current_epoch      = start_epoch
    training_exception = None

    try:
        for epoch in range(start_epoch, args.epochs):
            current_epoch = epoch
            print(f"\n{'═'*80}")
            print(f"  Epoch {epoch+1}/{args.epochs} | Step {global_step}/{total_steps} | "
                  f"BestValLoss={best_val_loss:.4f}")
            print(f"{'═'*80}")

            # [v13-FIX-C] epoch_ce / epoch_acc removed — derived from epoch_lang raw sums
            # epoch_lang[lang] = [sum_raw_ce, sum_correct_tokens, sum_valid_tokens]
            epoch_batches = 0
            accum_counter = 0
            epoch_lang: Dict[str, List] = defaultdict(lambda: [0.0, 0.0, 0.0])

            for batch in train_loader:
                if batch is None:
                    continue

                codes      = batch["codes"].to(device, non_blocking=True)
                phones     = batch["phones"].to(device, non_blocking=True)
                x_mask     = batch["x_mask"].to(device, non_blocking=True)
                phone_mask = batch["phone_mask"].to(device, non_blocking=True)
                languages  = batch["languages"]

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    logits, final_mask, x0 = _model_forward(
                        model, codes, x_mask, phones, phone_mask
                    )
                    # ce_all/mask_f are detached [v12-FIX-A]; graph freed after backward
                    ce_loss, ce_all, mask_f = _masked_ce_loss_with_raw(
                        logits, final_mask, x0
                    )
                    ewc_loss = (ewc.penalty(model)
                                if ewc is not None
                                else torch.tensor(0.0, device=device))
                    loss = (ce_loss + ewc_loss) / args.grad_accumulation

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_counter += 1

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
                    global_step  += 1
                    accum_counter = 0

                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        acc = compute_accuracy(logits, final_mask, x0)

                    epoch_batches += 1

                    # Accumulate raw token sums for epoch summary [v13-FIX-C]
                    _accum_epoch_lang_from_raw(
                        ce_all, mask_f, logits, final_mask, x0,
                        languages, device, epoch_lang
                    )

                    # ── Logging ───────────────────────────────────────────
                    if global_step % args.log_interval == 0:
                        # per_lang_metrics only at log steps — correct token-avg for this batch
                        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                            lm = per_lang_metrics(logits, final_mask, x0, languages, device)

                        lrs   = scheduler.get_last_lr()
                        lr_bb = lrs[0]
                        lr_pe = lrs[1] if len(lrs) > 1 else lrs[0]
                        gn    = (grad_norm.item()
                                 if isinstance(grad_norm, torch.Tensor)
                                 else float(grad_norm))
                        print(
                            f"  Step {global_step:6d}/{total_steps} | "
                            f"CE={ce_loss.item():.4f} | Acc={acc:.4f} | "
                            f"LR_bb={lr_bb:.2e} | LR_pe={lr_pe:.2e} | GNorm={gn:.3f}"
                            + (f" | EWC={ewc_loss.item():.4f}" if ewc else "")
                        )
                        lang_parts = [
                            f"{l}:[loss={lm[l]['loss']:.4f} acc={lm[l]['acc']:.4f}]"
                            for l in ALL_LANGUAGES if l in lm
                        ]
                        if lang_parts:
                            print("  langs | " + " ".join(lang_parts))

                        writer.add_scalar("Train/loss_ce",   ce_loss.item(), global_step)
                        writer.add_scalar("Train/top1_acc",  acc,            global_step)
                        writer.add_scalar("Train/grad_norm", gn,             global_step)
                        writer.add_scalar("LR/backbone",     lr_bb,          global_step)
                        writer.add_scalar("LR/phone_emb",    lr_pe,          global_step)
                        if ewc:
                            writer.add_scalar("Train/loss_ewc",   ewc_loss.item(), global_step)
                            writer.add_scalar("Train/loss_total",
                                              ce_loss.item() + ewc_loss.item(), global_step)
                        for lang, m in lm.items():
                            writer.add_scalar(f"Train/loss_{lang}", m["loss"], global_step)
                            writer.add_scalar(f"Train/acc_{lang}",  m["acc"],  global_step)

                    # ── Validation ────────────────────────────────────────
                    if args.val_interval > 0 and global_step % args.val_interval == 0:
                        val_metrics = evaluate(
                            model, val_loader, device, global_step,
                            use_amp=use_amp, amp_dtype=amp_dtype,
                        )
                        for k, v in val_metrics.items():
                            writer.add_scalar(f"Val/{k}", v, global_step)
                        if val_metrics["loss"] < best_val_loss:
                            best_val_loss = val_metrics["loss"]
                            save_checkpoint(
                                output_dir / "best_model.pth",
                                model, optimizer, scheduler, scaler,
                                epoch, global_step, best_val_loss, recent_ckpts
                            )
                            print(f"  [Val] ✓ New best: loss={best_val_loss:.4f}")

                    # ── Periodic checkpoint ───────────────────────────────
                    if global_step % args.save_every == 0:
                        ckpt_path = output_dir / f"step_{global_step:07d}.pth"
                        recent_ckpts.append(str(ckpt_path))
                        recent_ckpts = prune_old_checkpoints(recent_ckpts, args.keep_last)
                        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler,
                                        epoch, global_step, best_val_loss, recent_ckpts)
                        print(f"  [Ckpt] Saved {ckpt_path.name}")

                    if args.max_steps > 0 and global_step >= args.max_steps:
                        print("[Main] max_steps reached.")
                        break

            # ── End-of-epoch partial-window step ──────────────────────────────
            if accum_counter > 0:
                scale_factor = args.grad_accumulation / accum_counter
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(scale_factor)

                if scaler is not None:
                    scaler.unscale_(optimizer)
                partial_gn = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                global_step += 1

                gn_val = (partial_gn.item()
                          if isinstance(partial_gn, torch.Tensor)
                          else float(partial_gn))
                writer.add_scalar("Train/grad_norm_partial", gn_val, global_step)
                print(
                    f"  [Epoch {epoch+1}] Partial-window step "
                    f"({accum_counter}/{args.grad_accumulation} microbatches) "
                    f"GNorm={gn_val:.3f}"
                )

                if args.max_steps > 0 and global_step >= args.max_steps:
                    print("[Main] max_steps reached (partial-window step).")
                    optimizer.zero_grad(set_to_none=True)
                    break

            optimizer.zero_grad(set_to_none=True)

            # ── Epoch summary [v13-FIX-C] ─────────────────────────────────────
            # Derive overall metrics from raw token sums — token-weighted averages
            if epoch_batches > 0:
                total_sum_ce      = sum(v[0] for v in epoch_lang.values())
                total_sum_correct = sum(v[1] for v in epoch_lang.values())
                total_sum_m       = sum(v[2] for v in epoch_lang.values())
                avg_ce  = total_sum_ce      / max(total_sum_m, 1.0)
                avg_acc = total_sum_correct / max(total_sum_m, 1.0)
                writer.add_scalar("Epoch/train_loss", avg_ce,  epoch)
                writer.add_scalar("Epoch/train_acc",  avg_acc, epoch)
                print(f"\n  [Epoch {epoch+1}] AvgCE={avg_ce:.4f} | "
                      f"AvgAcc={avg_acc:.4f} | Steps={epoch_batches}")
                for lang in ALL_LANGUAGES:
                    if lang in epoch_lang and epoch_lang[lang][2] > 0:
                        sm = epoch_lang[lang][2]
                        el = epoch_lang[lang][0] / sm
                        ea = epoch_lang[lang][1] / sm
                        print(f"    [{lang}] loss={el:.4f} acc={ea:.4f}")
                        writer.add_scalar(f"Epoch/loss_{lang}", el, epoch)
                        writer.add_scalar(f"Epoch/acc_{lang}",  ea, epoch)

            # ── End-of-epoch validation ────────────────────────────────────────
            if args.val_interval == 0:
                val_metrics = evaluate(
                    model, val_loader, device, global_step,
                    use_amp=use_amp, amp_dtype=amp_dtype,
                )
                for k, v in val_metrics.items():
                    writer.add_scalar(f"Epoch/val_{k}", v, epoch)
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    save_checkpoint(
                        output_dir / "best_model.pth",
                        model, optimizer, scheduler, scaler,
                        epoch, global_step, best_val_loss, recent_ckpts
                    )
                    print(f"  [Val] ✓ New best: loss={best_val_loss:.4f}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt — saving emergency checkpoint.")
        training_exception = "KeyboardInterrupt"
    except Exception as e:
        print("\n[Main] Exception — saving emergency checkpoint.")
        traceback.print_exc()
        training_exception = str(e)
    finally:
        if training_exception is not None:
            emerg_path = output_dir / f"emergency_step{global_step}.pth"
            save_checkpoint(
                emerg_path, model, optimizer, scheduler, scaler,
                current_epoch, global_step, best_val_loss, recent_ckpts
            )
            print(f"[Main] Emergency checkpoint: {emerg_path.name}")

        hook.remove()
        print("[Main] Backward hook removed.")

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = output_dir / f"final_epoch{args.epochs}_step{global_step}.pth"
    recent_ckpts.append(str(final_path))
    recent_ckpts = prune_old_checkpoints(recent_ckpts, args.keep_last)
    save_checkpoint(final_path, model, optimizer, scheduler, scaler,
                    args.epochs, global_step, best_val_loss, recent_ckpts)

    print(f"\n[Main] Done. Final: {final_path.name} | BestValLoss={best_val_loss:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
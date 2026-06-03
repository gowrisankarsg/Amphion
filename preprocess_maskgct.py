#!/usr/bin/env python3
"""
preprocess_maskgct.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single-script pipeline for MaskGCT T2S fine-tuning.

Does three things in one run:
  1. PHONES  — chn_eng_tam_g2p(text) → saves phones/<audioname>.npy
  2. CODES   — W2V-BERT-2.0 layer 17 + MaskGCT codec → saves codes/<audioname>.npy
  3. MANIFEST — writes train/val JSONL under output-dir/

KEY DESIGN
━━━━━━━━━━
  • .npy filenames are always the AUDIO STEM (Path(audio).stem)
      e.g. audio/ISTL_0000202_0000009.wav  →  ISTL_0000202_0000009.npy
  • g2p_ok dict is keyed by STEM — no speaker-id collisions
  • "id" field in output JSONL comes from the source record's
      id / speaker / speaker_id field (for human reference only)

OUTPUT LAYOUT
━━━━━━━━━━━━━
  output-dir/
    ta/
      codes/   <audioname>.npy   int32 (T,)   values [0, 8191]
      phones/  <audioname>.npy   int32 (L,)   values [0, 1022]
    en/  ...
    zh/  ...
    ta_train.jsonl
    ta_val.jsonl

INPUT MANIFEST (one JSON per line):
  {
    "audio_path": "audio/ISTL_0000202_0000009.wav",
    "text":       "வணக்கம்",
    "language":   "ta",
    "speaker_id": "ISTL_0000202"
  }
  Accepted field aliases:
    audio    : "audio"      | "audio_path"
    id       : "id"         | "speaker"    | "speaker_id"   (JSONL id only)
    language : "language"   (falls back to ::lang suffix or --language)

EXAMPLE
━━━━━━━
  python preprocess_maskgct.py \
    --input-manifest /data/metadata.jsonl::ta \
    --output-dir     /data/maskgct_processed \
    --audio-root     /data/corpus \
    --semantic-stats checkpoints/wav2vec2bert_stats.pt \
    --codec-ckpt     checkpoints/semantic_codec/model.safetensors \
    --batch-size 32 --workers 4 --val-ratio 0.01
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import SeamlessM4TFeatureExtractor

import safetensors.torch
from huggingface_hub import hf_hub_download

from models.tts.maskgct.maskgct_utils import (
    build_semantic_codec,
    build_semantic_model,
    load_config,
)
from models.tts.maskgct.g2p.g2p_generation import chn_eng_tam_g2p


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MaskGCT preprocessing: phones + semantic codes + manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-manifest", dest="input_manifests", action="append", required=True,
        metavar="PATH[::lang]",
        help=(
            "Input JSONL manifest, optionally suffixed with ::lang. "
            "Accepted fields per record: audio/audio_path, text, "
            "language (optional), id/speaker/speaker_id (optional). "
            "Repeat flag for multiple languages."
        ),
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help=(
            "Single output root. .npy files go to output-dir/<lang>/codes/ "
            "and output-dir/<lang>/phones/. JSONL manifests go to output-dir/."
        ),
    )
    p.add_argument(
        "--audio-root", type=Path, default=None,
        help=(
            "Base directory prepended to relative audio paths in the manifest. "
            "Falls back to manifest parent dir, then cwd if omitted."
        ),
    )
    p.add_argument(
        "--semantic-stats", type=Path,
        default=Path("checkpoints/wav2vec2bert_stats.pt"),
        help="wav2vec2bert_stats.pt — mean/std for W2V-BERT layer-17 normalisation.",
    )
    p.add_argument(
        "--codec-ckpt", type=Path, default=None,
        help=(
            "MaskGCT semantic codec safetensors checkpoint. "
            "Auto-downloaded from amphion/MaskGCT if the file is missing."
        ),
    )
    p.add_argument(
        "--language", type=str, default=None,
        help="Fallback language (ta/en/zh) when not in the record or ::lang suffix.",
    )
    p.add_argument("--batch-size",       type=int,   default=16)
    p.add_argument("--workers",          type=int,   default=4)
    p.add_argument("--val-ratio",        type=float, default=0.01)
    p.add_argument("--max-duration-sec", type=float, default=20.0)
    p.add_argument("--min-duration-sec", type=float, default=0.5)
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip utterances where both .npy files already exist on disk.",
    )
    p.add_argument("--max-samples", type=int, default=0,
                   help="Hard cap per manifest (0 = all). Useful for dry runs.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Record field helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_audio_field(rec: dict) -> str:
    """Returns the audio path string, checking 'audio' then 'audio_path'."""
    return rec.get("audio") or rec.get("audio_path") or ""


def get_stem(rec: dict) -> str:
    """
    Derives the .npy filename stem from the audio path.
    e.g. "audio/ISTL_0000202_0000009.wav"  →  "ISTL_0000202_0000009"
    This is the PRIMARY key used for all file I/O — never the speaker id.
    """
    return Path(get_audio_field(rec)).stem


def get_uid(rec: dict) -> str:
    """
    Returns the value written to the "id" field in output JSONL only.
    Tries: 'id' → 'speaker' → 'speaker_id' → audio stem (last resort).
    Never used as a dict key for g2p_ok or for .npy filenames.
    """
    return (
        rec.get("id")
        or rec.get("speaker")
        or rec.get("speaker_id")
        or get_stem(rec)
    )


def has_required_fields(rec: dict, lineno: int) -> bool:
    """Validates that both an audio field and text field exist."""
    if not get_audio_field(rec):
        print(f"  [Skip] line {lineno}: missing 'audio' / 'audio_path' field")
        return False
    if not rec.get("text", "").strip():
        print(f"  [Skip] line {lineno}: missing or empty 'text' field")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Misc helpers
# ══════════════════════════════════════════════════════════════════════════════

def parse_manifest_specs(entries: List[str]) -> List[Tuple[Path, Optional[str]]]:
    result = []
    for raw in entries:
        raw  = raw.strip()
        lang: Optional[str] = None
        if "::" in raw:
            path_str, lang_part = raw.rsplit("::", 1)
            raw  = path_str.strip()
            lang = lang_part.strip().lower() or None
        result.append((Path(raw).expanduser(), lang))
    return result


def resolve_audio(audio_field: str, manifest_dir: Path,
                  audio_root: Optional[Path]) -> Optional[Path]:
    """
    Resolution order:
      1. Absolute path as-is
      2. --audio-root / audio_field
      3. manifest parent dir / audio_field
      4. cwd / audio_field
    """
    candidates = [Path(audio_field).expanduser()]
    for base in filter(None, [audio_root, manifest_dir, Path(".").resolve()]):
        candidates.append((base / audio_field).expanduser())
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def assign_val(stem: str, ratio: float) -> bool:
    """Deterministic train/val split keyed on the audio stem."""
    if ratio <= 0.0:
        return False
    digest = hashlib.sha1(stem.encode()).hexdigest()
    return (int(digest, 16) % 1_000_000) / 1_000_000 < ratio


def load_audio_mono_16k(path: Path) -> Tuple[torch.Tensor, float]:
    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    duration = wav.shape[1] / sr
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    return wav, duration


def load_existing_stems(path: Path) -> set:
    """Reads existing JSONL and returns the set of audio stems already written."""
    stems: set = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    # stem is recoverable from the codes path: "codes/<stem>.npy"
                    codes_path = rec.get("semantic_codes_path", "")
                    if codes_path:
                        stems.add(Path(codes_path).stem)
                except (json.JSONDecodeError, KeyError):
                    pass
    return stems


# ══════════════════════════════════════════════════════════════════════════════
# Semantic extractor  (W2V-BERT-2.0 layer 17 + MaskGCT VQ codec)
# ══════════════════════════════════════════════════════════════════════════════

class SemanticExtractor:

    def __init__(self, stats_path: Path, codec_ckpt: Optional[Path],
                 device: torch.device) -> None:
        self.device = device

        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )

        self.semantic_model, self.semantic_mean, self.semantic_std = (
            build_semantic_model(device)
        )
        self.semantic_model = self.semantic_model.to(device).eval()
        self.semantic_mean  = self.semantic_mean.to(device)
        self.semantic_std   = self.semantic_std.to(device)

        cfg = load_config("./models/tts/maskgct/config/maskgct.json")
        self.semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)

        # Download if not given or file doesn't exist yet
        if codec_ckpt is None or not Path(codec_ckpt).exists():
            print("[Codec] File not found locally — downloading from amphion/MaskGCT ...")
            codec_ckpt = Path(hf_hub_download(
                "amphion/MaskGCT",
                filename="semantic_codec/model.safetensors",
            ))

        safetensors.torch.load_model(
            self.semantic_codec, str(codec_ckpt), device=str(device)
        )
        self.semantic_codec = self.semantic_codec.to(device).eval()
        print(f"[SemanticExtractor] Ready — device={device}")
        print(f"  stats : {stats_path}")
        print(f"  codec : {codec_ckpt}")

    @torch.inference_mode()
    def extract_batch(self, waveforms: List[torch.Tensor]) -> np.ndarray:
        """
        waveforms : list of (1, T) at 16 kHz
        returns   : int32 ndarray (B, T_codes), values in [0, 8191]
        """
        arrays = [w.squeeze(0).cpu().numpy() for w in waveforms]
        inputs = self.feature_extractor(
            arrays, sampling_rate=16_000, return_tensors="pt", padding=True,
        )
        input_features = inputs["input_features"].to(self.device)
        attention_mask  = inputs["attention_mask"].to(self.device)

        outputs = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = outputs.hidden_states[17]
        feat = (feat - self.semantic_mean) / self.semantic_std

        semantic_code, _ = self.semantic_codec.quantize(feat)
        if semantic_code.dim() == 1:
            semantic_code = semantic_code.unsqueeze(0)

        return semantic_code.detach().cpu().numpy().astype(np.int32)  # (B, T_codes)


# ══════════════════════════════════════════════════════════════════════════════
# G2P phone extractor
# ══════════════════════════════════════════════════════════════════════════════

class PhoneExtractor:
    """
    Thin wrapper around chn_eng_tam_g2p.
    Handles Tamil, English, Chinese (auto-detected per character).
    """

    def extract(self, text: str) -> Optional[np.ndarray]:
        try:
            _phoneme_str, token_ids = chn_eng_tam_g2p(text)
            if not token_ids:
                return None
            ids = np.array(token_ids, dtype=np.int32)
            if ids.max() > 1022 or ids.min() < 0:
                print(f"  [Warn G2P] ID out of range "
                      f"max={ids.max()} min={ids.min()} text='{text[:50]}'")
                return None
            return ids
        except Exception as e:
            print(f"  [G2P Error] text='{text[:50]}': {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
# Per-manifest processing
# ══════════════════════════════════════════════════════════════════════════════

def process_manifest(
    manifest_path: Path,
    manifest_lang: Optional[str],
    output_dir: Path,
    semantic_extractor: SemanticExtractor,
    phone_extractor: PhoneExtractor,
    args: argparse.Namespace,
) -> None:
    manifest_path = manifest_path.resolve()
    manifest_dir  = manifest_path.parent
    default_lang  = manifest_lang or args.language

    if default_lang is None:
        raise ValueError(
            f"Language unknown for '{manifest_path.name}'. "
            "Add a '::lang' suffix or use --language."
        )

    # ── Directory cache (lazy, per language) ─────────────────────────────────
    _lang_dirs: Dict[str, Tuple[Path, Path]] = {}

    def lang_dirs(lang: str) -> Tuple[Path, Path]:
        if lang not in _lang_dirs:
            cd = output_dir / lang / "codes"
            pd = output_dir / lang / "phones"
            cd.mkdir(parents=True, exist_ok=True)
            pd.mkdir(parents=True, exist_ok=True)
            _lang_dirs[lang] = (cd, pd)
        return _lang_dirs[lang]

    # ── Manifest file handles (lazy, per language) ────────────────────────────
    # Each entry: (train_fh, val_fh, train_stems: set, val_stems: set)
    _mhandles: Dict[str, Tuple] = {}

    def mhandles(lang: str) -> Tuple:
        if lang not in _mhandles:
            tp = output_dir / f"{lang}_train.jsonl"
            vp = output_dir / f"{lang}_val.jsonl"
            _mhandles[lang] = (
                open(tp, "a", encoding="utf-8"),
                open(vp, "a", encoding="utf-8"),
                load_existing_stems(tp),   # set of stems already in train JSONL
                load_existing_stems(vp),   # set of stems already in val JSONL
            )
        return _mhandles[lang]

    def write_manifest(lang: str, rec: dict, stem: str,
                       code_len: int, phone_len: int,
                       duration: float) -> Optional[str]:
        """
        Writes one entry to train or val JSONL.
        Returns "train", "val", or None (if stem was already written before).
        """
        train_f, val_f, train_stems, val_stems = mhandles(lang)
        if stem in train_stems or stem in val_stems:
            return None
        entry = {
            # "id" is for human reference — comes from source record's id/speaker fields
            "id":                  get_uid(rec),
            # paths relative to output-dir/<lang>/
            "semantic_codes_path": f"codes/{stem}.npy",
            "phone_ids_path":      f"phones/{stem}.npy",
            "language":            lang,
            "code_len":            code_len,
            "phone_len":           phone_len,
            # informational fields
            "text":                rec.get("text", ""),
            "audio":               get_audio_field(rec),
            "duration":            round(duration, 3),
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        if assign_val(stem, args.val_ratio):
            val_f.write(line);    val_stems.add(stem);   return "val"
        else:
            train_f.write(line);  train_stems.add(stem); return "train"

    # ── Read records ──────────────────────────────────────────────────────────
    records: List[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [Skip] line {lineno} JSON error: {e}")
                continue
            if has_required_fields(rec, lineno):
                records.append(rec)

    if args.max_samples > 0:
        records = records[: args.max_samples]

    print(f"\n[{default_lang}] {manifest_path.name} — {len(records):,} records")

    stats = dict(
        total=len(records), already_done=0,
        skipped_g2p=0, skipped_audio=0, skipped_duration=0, skipped_codes=0,
        written_train=0, written_val=0,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 1 — G2P  (CPU)
    # Key: stem (audio filename without extension) — unique per utterance
    # ══════════════════════════════════════════════════════════════════════════
    print("  [Pass 1/2] G2P → phones/<stem>.npy ...")

    # stem → phone_ids  (this is the only key used throughout)
    g2p_ok: Dict[str, np.ndarray] = {}

    for rec in tqdm(records, desc="  G2P", unit="utt", leave=False):
        stem = get_stem(rec)
        lang = rec.get("language", default_lang).lower()

        codes_dir, phones_dir = lang_dirs(lang)
        phones_path = phones_dir / f"{stem}.npy"
        codes_path  = codes_dir  / f"{stem}.npy"

        # Both files already on disk
        if args.skip_existing and codes_path.exists() and phones_path.exists():
            stats["already_done"] += 1
            g2p_ok[stem] = np.load(str(phones_path), allow_pickle=False)
            continue

        # Phones done, codes not yet — load phones, skip G2P
        if phones_path.exists() and args.skip_existing:
            g2p_ok[stem] = np.load(str(phones_path), allow_pickle=False)
            continue

        text = rec.get("text", "").strip()
        phone_ids = phone_extractor.extract(text)
        if phone_ids is None:
            stats["skipped_g2p"] += 1
            continue

        np.save(str(phones_path), phone_ids)
        g2p_ok[stem] = phone_ids

    n_written = len(g2p_ok) - stats["already_done"]
    print(f"  [Pass 1/2] done — written={n_written:,}  "
          f"already_done={stats['already_done']:,}  "
          f"skipped={stats['skipped_g2p']:,}")

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 2 — Semantic codes  (GPU, batched)
    # ══════════════════════════════════════════════════════════════════════════
    print("  [Pass 2/2] Semantic codes → codes/<stem>.npy ...")

    # Only records whose G2P succeeded
    work = [r for r in records if get_stem(r) in g2p_ok]

    batch_wavs: List[torch.Tensor] = []
    batch_recs: List[dict]         = []
    batch_durs: List[float]        = []

    def flush() -> None:
        if not batch_wavs:
            return
        try:
            codes_np = semantic_extractor.extract_batch(batch_wavs)  # (B, T_codes)
        except Exception as e:
            print(f"\n  [Error] GPU batch failed: {e}")
            stats["skipped_codes"] += len(batch_wavs)
            batch_wavs.clear(); batch_recs.clear(); batch_durs.clear()
            return

        for i, rec in enumerate(batch_recs):
            stem      = get_stem(rec)
            lang      = rec.get("language", default_lang).lower()
            phone_ids = g2p_ok[stem]

            codes_dir, _ = lang_dirs(lang)
            codes_path   = codes_dir / f"{stem}.npy"
            code_row     = codes_np[i]

            if code_row.size == 0:
                stats["skipped_codes"] += 1
                continue
            if code_row.max() > 8191 or code_row.min() < 0:
                print(f"\n  [Skip] Code out of range: stem={stem}")
                stats["skipped_codes"] += 1
                continue

            np.save(str(codes_path), code_row)

            split = write_manifest(
                lang, rec, stem,
                code_len=int(code_row.shape[0]),
                phone_len=int(phone_ids.shape[0]),
                duration=batch_durs[i],
            )
            if split == "train": stats["written_train"] += 1
            elif split == "val": stats["written_val"]   += 1

        batch_wavs.clear(); batch_recs.clear(); batch_durs.clear()

    for rec in tqdm(work, desc="  Codes", unit="utt", leave=False):
        stem = get_stem(rec)
        lang = rec.get("language", default_lang).lower()
        phone_ids = g2p_ok[stem]

        codes_dir, _ = lang_dirs(lang)
        codes_path   = codes_dir / f"{stem}.npy"

        # Codes already on disk — just write manifest entry
        if args.skip_existing and codes_path.exists():
            code_row = np.load(str(codes_path), allow_pickle=False)
            split = write_manifest(
                lang, rec, stem,
                code_len=int(code_row.shape[0]),
                phone_len=int(phone_ids.shape[0]),
                duration=rec.get("duration", 0.0),
            )
            if split == "train": stats["written_train"] += 1
            elif split == "val": stats["written_val"]   += 1
            continue

        # Resolve audio path
        audio_field = get_audio_field(rec)
        audio_path  = resolve_audio(audio_field, manifest_dir, args.audio_root)
        if audio_path is None:
            print(f"\n  [Skip] Audio not found: '{audio_field}'  stem={stem}")
            stats["skipped_audio"] += 1
            continue

        try:
            wav, duration = load_audio_mono_16k(audio_path)
        except Exception as e:
            print(f"\n  [Skip] Audio load error stem={stem}: {e}")
            stats["skipped_audio"] += 1
            continue

        if duration < args.min_duration_sec or duration > args.max_duration_sec:
            stats["skipped_duration"] += 1
            continue

        batch_wavs.append(wav)
        batch_recs.append(rec)
        batch_durs.append(duration)

        if len(batch_wavs) >= args.batch_size:
            flush()

    flush()  # remaining partial batch

    # ── Close manifest handles and print summary ──────────────────────────────
    for lang, (train_f, val_f, train_stems, val_stems) in _mhandles.items():
        train_f.close()
        val_f.close()
        print(f"\n  [{lang}]  train={len(train_stems):,}  val={len(val_stems):,}")
        print(f"    → {output_dir}/{lang}_train.jsonl")
        print(f"    → {output_dir}/{lang}_val.jsonl")

    print(f"\n  [Stats]")
    print(f"    total             : {stats['total']:>8,}")
    print(f"    already done      : {stats['already_done']:>8,}")
    print(f"    skipped (audio)   : {stats['skipped_audio']:>8,}")
    print(f"    skipped (duration): {stats['skipped_duration']:>8,}")
    print(f"    skipped (G2P)     : {stats['skipped_g2p']:>8,}")
    print(f"    skipped (codes)   : {stats['skipped_codes']:>8,}")
    print(f"    written train     : {stats['written_train']:>8,}")
    print(f"    written val       : {stats['written_val']:>8,}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess_maskgct] device={device}")
    print(f"  output-dir : {out}")

    manifest_specs = parse_manifest_specs(args.input_manifests)

    print("\n[Models] Loading W2V-BERT + MaskGCT semantic codec ...")
    semantic_extractor = SemanticExtractor(
        stats_path = args.semantic_stats.expanduser().resolve(),
        codec_ckpt = args.codec_ckpt.expanduser().resolve() if args.codec_ckpt else None,
        device     = device,
    )

    print("[Models] PhonemeBpeTokenizer loaded via chn_eng_tam_g2p import.")
    phone_extractor = PhoneExtractor()

    for manifest_path, manifest_lang in manifest_specs:
        if not manifest_path.exists():
            print(f"\n[Skip] Manifest not found: {manifest_path}")
            continue
        process_manifest(
            manifest_path      = manifest_path,
            manifest_lang      = manifest_lang,
            output_dir         = out,
            semantic_extractor = semantic_extractor,
            phone_extractor    = phone_extractor,
            args               = args,
        )

    print("\n[Done] Training command:")
    for _, mlang in manifest_specs:
        lang = mlang or args.language or "??"
        print(f"  --train-manifest {out}/{lang}_train.jsonl::{lang} \\")
    for _, mlang in manifest_specs:
        lang = mlang or args.language or "??"
        print(f"  --val-manifest   {out}/{lang}_val.jsonl::{lang} \\")
    print(f"  --root-dir {out}")


if __name__ == "__main__":
    main()

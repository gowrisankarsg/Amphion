#!/usr/bin/env python3
"""
preprocess_maskgct.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single-script pipeline for MaskGCT T2S fine-tuning.

Does three things in one run:
  1. PHONES  — chn_eng_tam_g2p(text) → saves phones/<audioname>.npy
  2. CODES   — W2V-BERT-2.0 layer 17 + MaskGCT codec → saves codes/<audioname>.npy
  3. MANIFEST — writes train/val JSONL under output-dir/

OUTPUT LAYOUT (single tree, --output-dir is both root-dir and manifest dir)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  output-dir/
    ta/
      codes/   <audioname>.npy   int32 (T,)  values [0, 8191]
      phones/  <audioname>.npy   int32 (L,)  values [0, 1022]
    en/
      codes/   ...
      phones/  ...
    zh/
      codes/   ...
      phones/  ...
    ta_train.jsonl
    ta_val.jsonl
    en_train.jsonl  ...

INPUT MANIFEST (one JSON per line):
  {"id": "ta_0001", "audio": "wavs/ta_0001.wav", "text": "வணக்கம்", "language": "ta"}

  The .npy stem is taken from the AUDIO filename (Path(audio).stem),
  NOT from the "id" field. "id" is only used for deduplication in manifests.

EXAMPLE
━━━━━━━
  # Tamil only
  python preprocess_maskgct.py \
    --input-manifest data/ta_raw.jsonl::ta \
    --output-dir     /data/maskgct_processed \
    --semantic-stats checkpoints/wav2vec2bert_stats.pt \
    --codec-ckpt     checkpoints/semantic_codec/model.safetensors \
    --batch-size 16 --workers 4 --val-ratio 0.01

  # All three languages
  python preprocess_maskgct.py \
    --input-manifest data/ta_raw.jsonl::ta \
    --input-manifest data/en_raw.jsonl::en \
    --input-manifest data/zh_raw.jsonl::zh \
    --output-dir     /data/maskgct_processed \
    --semantic-stats checkpoints/wav2vec2bert_stats.pt \
    --codec-ckpt     checkpoints/semantic_codec/model.safetensors \
    --batch-size 32 --workers 8 --val-ratio 0.01

  # Training command:
  python train_maskgct_t2s.py \
    --root-dir       /data/maskgct_processed \
    --train-manifest /data/maskgct_processed/ta_train.jsonl::ta \
    --val-manifest   /data/maskgct_processed/ta_val.jsonl::ta
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

from models.tts.maskgct.maskgct_utils import build_semantic_codec, build_semantic_model,load_config
# chn_eng_tam_g2p handles Tamil + English + Chinese in one call.
# It is the module-level function in g2p_generation.py — importing it also
# triggers the module-level PhonemeBpeTokenizer + vocab.json load at the
# bottom of that file, so no separate initialisation is needed here.
from models.tts.maskgct.g2p.g2p_generation import chn_eng_tam_g2p


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MaskGCT preprocessing: phones + semantic codes + train/val manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--input-manifest", dest="input_manifests", action="append", required=True,
        metavar="PATH[::lang]",
        help=(
            "Input JSONL manifest. Optionally suffix with ::lang to set the "
            "default language for all records in that file. "
            "Each line must have fields: id, audio, text. "
            "A per-record 'language' field overrides the ::lang suffix. "
            "Repeat for multiple languages."
        ),
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help=(
            "Single output root. All .npy files and JSONL manifests are "
            "written here. Layout: output-dir/<lang>/codes/ and phones/, "
            "plus output-dir/<lang>_train.jsonl / <lang>_val.jsonl."
        ),
    )
    p.add_argument(
        "--audio-root", type=Path, default=None,
        help=(
            "Base directory prepended to relative audio paths. "
            "If omitted, paths are resolved relative to the manifest file's "
            "parent, then the current working directory."
        ),
    )
    p.add_argument(
        "--semantic-stats", type=Path,
        default=Path("checkpoints/wav2vec2bert_stats.pt"),
        help="wav2vec2bert_stats.pt — mean/std tensors for W2V-BERT layer-17 normalisation.",
    )
    p.add_argument(
        "--codec-ckpt", type=Path, default=None,
        help=(
            "MaskGCT semantic codec safetensors checkpoint. "
            "Downloaded from HuggingFace amphion/MaskGCT if omitted."
        ),
    )
    p.add_argument(
        "--language", type=str, default=None,
        help=(
            "Fallback language code (ta/en/zh) when neither the record's "
            "'language' field nor a ::lang suffix is present."
        ),
    )
    p.add_argument("--batch-size",       type=int,   default=16)
    p.add_argument("--workers",          type=int,   default=4,
                   help="CPU threads for parallel audio loading.")
    p.add_argument("--val-ratio",        type=float, default=0.01,
                   help="Validation split fraction (deterministic SHA1 hash).")
    p.add_argument("--max-duration-sec", type=float, default=20.0,
                   help="Drop audio longer than this to protect GPU VRAM.")
    p.add_argument("--min-duration-sec", type=float, default=0.5,
                   help="Drop audio shorter than this.")
    p.add_argument("--skip-existing",    action="store_true",
                   help="Skip utterances where both .npy files already exist.")
    p.add_argument("--max-samples",      type=int,   default=0,
                   help="Hard cap per manifest (0 = all). Useful for dry runs.")
    p.add_argument("--device",           type=str,   default="cuda")
    p.add_argument("--seed",             type=int,   default=42)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
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
      2. --audio-root / audio_field   (if --audio-root given)
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


def audio_stem(audio_field: str) -> str:
    """
    Extract the filename stem from an audio path string.
    e.g. "wavs/ta_0001.wav"  →  "ta_0001"
         "/data/audio.flac"  →  "audio"
    """
    return Path(audio_field).stem


def assign_val(uid: str, ratio: float) -> bool:
    """Deterministic train/val assignment — same uid always lands in the same split."""
    if ratio <= 0.0:
        return False
    digest = hashlib.sha1(uid.encode()).hexdigest()
    return (int(digest, 16) % 1_000_000) / 1_000_000 < ratio


def load_audio_mono_16k(path: Path) -> Tuple[torch.Tensor, float]:
    """
    Load → mono → resample to 16 kHz (required by W2V-BERT).
    Returns (waveform [1, T_16k], duration_seconds_at_original_sr).
    Duration is computed BEFORE resampling so it reflects real audio length.
    """
    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    duration = wav.shape[1] / sr          # seconds, based on original sr
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    return wav, duration


def load_existing_ids(path: Path) -> set:
    ids: set = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return ids


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
            build_semantic_model(self.device)
        )
        self.semantic_model = self.semantic_model.to(device).eval()
        self.semantic_mean  = self.semantic_mean.to(device)
        self.semantic_std   = self.semantic_std.to(device)

        self.cfg = load_config("./models/tts/maskgct/config/maskgct.json")
        self.semantic_codec = build_semantic_codec(
          self.cfg.model.semantic_codec,
          self.device
        )
        if codec_ckpt is None:
            print("[Codec] Downloading amphion/MaskGCT semantic_codec from HuggingFace ...")
            codec_ckpt = Path(hf_hub_download(
                "amphion/MaskGCT",
                filename="semantic_codec/model.safetensors",
            ))
        safetensors.torch.load_model(self.semantic_codec, str(codec_ckpt))
        self.semantic_codec = self.semantic_codec.to(device).eval()

        print(f"[SemanticExtractor] Ready — device={device}")

    @torch.inference_mode()
    def extract_batch(self, waveforms: List[torch.Tensor]) -> np.ndarray:
        """
        waveforms : list of (1, T) tensors at 16 kHz
        returns   : int32 ndarray shape (B, T_codes), values in [0, 8191]

        W2V-BERT pads the batch to the longest waveform, so grouping
        similar-duration utterances into one batch reduces wasted compute.
        """
        arrays = [w.squeeze(0).cpu().numpy() for w in waveforms]

        inputs = self.feature_extractor(
            arrays,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs["input_features"].to(self.device)
        attention_mask  = inputs["attention_mask"].to(self.device)

        outputs = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        feat = outputs.hidden_states[17]                          # layer 17
        feat = (feat - self.semantic_mean) / self.semantic_std    # normalise

        semantic_code, _ = self.semantic_codec.quantize(feat)
        if semantic_code.dim() == 1:
            semantic_code = semantic_code.unsqueeze(0)

        return semantic_code.detach().cpu().numpy().astype(np.int32)  # (B, T_codes)


# ══════════════════════════════════════════════════════════════════════════════
# G2P phone extractor  —  wraps chn_eng_tam_g2p
# ══════════════════════════════════════════════════════════════════════════════

class PhoneExtractor:
    """
    Calls chn_eng_tam_g2p(text) which handles Tamil, English, and Chinese
    including code-mixed text in a single pass.

    chn_eng_tam_g2p returns (phoneme_str, token_ids):
      phoneme_str : pipe-separated IPA string  e.g. "ʋ|a|ɳ|a|k|k|a|m"
      token_ids   : list[int] — vocab IDs, all in [0, 1022]

    The 'language' argument to extract() is accepted for API consistency
    but not used — chn_eng_tam_g2p auto-detects script per character.
    """

    def extract(self, text: str, language: str = "") -> Optional[np.ndarray]:
        try:
            _phoneme_str, token_ids = chn_eng_tam_g2p(text)
            if not token_ids:
                return None
            ids = np.array(token_ids, dtype=np.int32)
            if ids.max() > 1022 or ids.min() < 0:
                print(f"  [Warn G2P] Phone ID out of range "
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
    """
    Full pipeline for one input JSONL manifest:
      Pass 1 (CPU)  — G2P all texts → phones/<audioname>.npy
      Pass 2 (GPU)  — extract semantic codes → codes/<audioname>.npy
      Writes        — <lang>_train.jsonl / <lang>_val.jsonl
    """
    manifest_path = manifest_path.resolve()
    manifest_dir  = manifest_path.parent
    default_lang  = manifest_lang or args.language

    if default_lang is None:
        raise ValueError(
            f"Language unknown for manifest '{manifest_path.name}'. "
            "Use a '::lang' suffix or --language."
        )

    # ── Directory cache  (created lazily per language) ────────────────────────
    _lang_dirs: Dict[str, Tuple[Path, Path]] = {}

    def lang_dirs(lang: str) -> Tuple[Path, Path]:
        if lang not in _lang_dirs:
            cd = output_dir / lang / "codes"
            pd = output_dir / lang / "phones"
            cd.mkdir(parents=True, exist_ok=True)
            pd.mkdir(parents=True, exist_ok=True)
            _lang_dirs[lang] = (cd, pd)
        return _lang_dirs[lang]

    # ── Manifest file handles  (opened lazily per language) ───────────────────
    # Each entry: (train_fh, val_fh, train_ids: set, val_ids: set)
    _mhandles: Dict[str, Tuple] = {}

    def mhandles(lang: str) -> Tuple:
        if lang not in _mhandles:
            tp = output_dir / f"{lang}_train.jsonl"
            vp = output_dir / f"{lang}_val.jsonl"
            _mhandles[lang] = (
                open(tp, "a", encoding="utf-8"),
                open(vp, "a", encoding="utf-8"),
                load_existing_ids(tp),
                load_existing_ids(vp),
            )
        return _mhandles[lang]

    def write_manifest(lang: str, rec: dict,
                       code_len: int, phone_len: int,
                       stem: str, duration: float) -> None:
        train_f, val_f, train_ids, val_ids = mhandles(lang)
        uid = rec["id"]
        if not uid:
          uid = rec.get("speaker", "speaker_id")
        if uid in train_ids or uid in val_ids:
            return  # already written in a previous run
        entry = {
            "id":                  uid,
            # Paths are relative to output-dir/<lang>/
            # so train_maskgct_t2s can resolve them as root_dir/lang/path
            "semantic_codes_path": f"codes/{stem}.npy",
            "phone_ids_path":      f"phones/{stem}.npy",
            "language":            lang,
            "code_len":            code_len,
            "phone_len":           phone_len,
            # informational — not used by the trainer
            "text":                rec.get("text", ""),
            "audio":               rec.get("audio", "audio_path"),
            "duration":            round(duration, 3),
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        if assign_val(uid, args.val_ratio):
            val_f.write(line);   val_ids.add(uid)
        else:
            train_f.write(line); train_ids.add(uid)

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
            required_groups = [
                ("id", "speaker", "speaker_id"),
                ("audio", "audio_path"),
                ("text",)
            ]
            
            for fields in required_groups:
                if not any(field in rec for field in fields):
                    print(f"  [Skip] line {lineno}: missing one of {fields}")
                    break
            else:
                records.append(rec)

    if args.max_samples > 0:
        records = records[: args.max_samples]

    print(f"\n[{default_lang}] {manifest_path.name} — {len(records):,} records")

    stats = dict(total=len(records), already_done=0,
                 skipped_g2p=0, skipped_audio=0, skipped_duration=0,
                 skipped_codes=0, written_train=0, written_val=0)

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 1 — G2P  (CPU only)
    # ══════════════════════════════════════════════════════════════════════════
    print("  [Pass 1/2] G2P → phones/<audioname>.npy ...")

    # uid → (stem, phone_ids_ndarray)
    g2p_ok: Dict[str, Tuple[str, np.ndarray]] = {}

    for rec in tqdm(records, desc="  G2P", unit="utt", leave=False):
        uid   = rec["id"]
        if not uid:
          uid = rec.get("speaker", "speaker_id")
        lang  = rec.get("language", default_lang).lower()
        stem  = audio_stem(rec.get("audio","audio_path"))          # ← filename stem of audio

        codes_dir, phones_dir = lang_dirs(lang)
        phones_path = phones_dir / f"{stem}.npy"
        codes_path  = codes_dir  / f"{stem}.npy"

        # Both outputs already on disk — nothing to do for this record
        if args.skip_existing and codes_path.exists() and phones_path.exists():
            stats["already_done"] += 1
            # Still need the phone array for manifest writing later
            g2p_ok[uid] = (stem, np.load(str(phones_path), allow_pickle=False))
            continue

        # Phones already on disk (codes not yet) — load instead of re-running G2P
        if phones_path.exists() and args.skip_existing:
            g2p_ok[uid] = (stem, np.load(str(phones_path), allow_pickle=False))
            continue

        text = rec.get("text", "").strip()
        if not text:
            stats["skipped_g2p"] += 1
            continue

        phone_ids = phone_extractor.extract(text, lang)
        if phone_ids is None:
            stats["skipped_g2p"] += 1
            continue

        np.save(str(phones_path), phone_ids)
        g2p_ok[uid] = (stem, phone_ids)

    n_g2p = len(g2p_ok) - stats["already_done"]
    print(f"  [Pass 1/2] done — G2P written={n_g2p:,} "
          f"already_done={stats['already_done']:,} "
          f"skipped={stats['skipped_g2p']:,}")

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 2 — Semantic codes  (GPU, batched)
    # ══════════════════════════════════════════════════════════════════════════
    print("  [Pass 2/2] Semantic codes → codes/<audioname>.npy ...")

    # Only process records that have valid phone IDs
    work = [r for r in records if r["id"] in g2p_ok]

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
            uid  = rec["id"]
            if not uid:
              uid = rec.get("speaker", "speaker_id")
            lang = rec.get("language", default_lang).lower()
            stem, phone_ids = g2p_ok[uid]
            codes_dir, _ = lang_dirs(lang)
            codes_path   = codes_dir / f"{stem}.npy"

            code_row = codes_np[i]  # (T_codes,)

            if code_row.size == 0:
                stats["skipped_codes"] += 1
                continue
            if code_row.max() > 8191 or code_row.min() < 0:
                print(f"\n  [Skip] Code out of range: {uid} stem={stem}")
                stats["skipped_codes"] += 1
                continue

            np.save(str(codes_path), code_row)

            # Manifest
            before_t = stats["written_train"]
            before_v = stats["written_val"]
            write_manifest(lang, rec,
                           code_len=int(code_row.shape[0]),
                           phone_len=int(phone_ids.shape[0]),
                           stem=stem,
                           duration=batch_durs[i])
            # Update written counters (write_manifest touches the sets directly)
            _, _, train_ids, val_ids = mhandles(lang)
            if uid in train_ids:
                stats["written_train"] += 1
            elif uid in val_ids:
                stats["written_val"] += 1

        batch_wavs.clear(); batch_recs.clear(); batch_durs.clear()

    for rec in tqdm(work, desc="  Codes", unit="utt", leave=False):
        uid  = rec["id"]
        if not uid:
          uid = rec.get("speaker", "speaker_id")
        lang = rec.get("language", default_lang).lower()
        stem, phone_ids = g2p_ok[uid]
        codes_dir, _ = lang_dirs(lang)
        codes_path   = codes_dir / f"{stem}.npy"

        # Codes already on disk — just write manifest entry and move on
        if args.skip_existing and codes_path.exists():
            code_row = np.load(str(codes_path), allow_pickle=False)
            write_manifest(lang, rec,
                           code_len=int(code_row.shape[0]),
                           phone_len=int(phone_ids.shape[0]),
                           stem=stem,
                           duration=rec.get("duration", 0.0))
            continue

        # Resolve audio
        audio_field = rec.get("audio", "audio_path")
        audio_path  = resolve_audio(audio_field, manifest_dir, args.audio_root)
        if audio_path is None:
            print(f"\n  [Skip] Audio not found: '{audio_field}'  id={uid}")
            stats["skipped_audio"] += 1
            continue

        try:
            wav, duration = load_audio_mono_16k(audio_path)
        except Exception as e:
            print(f"\n  [Skip] Audio load error id={uid}: {e}")
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

    # ── Close manifest file handles ───────────────────────────────────────────
    for lang, (train_f, val_f, train_ids, val_ids) in _mhandles.items():
        train_f.close()
        val_f.close()
        print(f"\n  [{lang}]  train={len(train_ids):,}  val={len(val_ids):,}")
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

    # Models are loaded ONCE and shared across all languages
    print("\n[Models] Loading W2V-BERT + MaskGCT semantic codec ...")
    semantic_extractor = SemanticExtractor(
        stats_path = args.semantic_stats.expanduser().resolve(),
        codec_ckpt = args.codec_ckpt.expanduser().resolve() if args.codec_ckpt else None,
        device     = device,
    )

    # chn_eng_tam_g2p is already initialised at import time (module-level in
    # g2p_generation.py).  PhoneExtractor is just a thin wrapper.
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

    # ── Print final training command ──────────────────────────────────────────
    print("\n[Done] Training command:")
    train_lines = []
    val_lines   = []
    for _, mlang in manifest_specs:
        lang = mlang or args.language or "??"
        train_lines.append(f"  --train-manifest {out}/{lang}_train.jsonl::{lang} \\")
        val_lines.append(  f"  --val-manifest   {out}/{lang}_val.jsonl::{lang} \\")
    print(f"python train_maskgct_t2s.py \\")
    print(f"  --root-dir {out} \\")
    for line in train_lines + val_lines:
        print(line)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Compute CLIP image-text similarity for existing sweep results from folder structure,
with incremental computation.

This script DOES NOT rely on metadata_all_both.csv or metadata_sweep.csv.

It scans sweep outputs like:
    sweep/<experiment>/<ckpt_tag>/<target_dim>/<image_id>/<image_id>_<target_dim>_<target_level>.png

and tries to read the full prompt from:
    sweep/<experiment>/<ckpt_tag>/prompts/<image_id>_<target_dim>_<target_level>.txt

If the full prompt is missing, the script still computes target_short, target_static,
and low/medium/high contrastive CLIP scores.

Dependencies:
  pip install transformers pillow pandas tqdm
"""

from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# ============================================================
# 0. USER SETTINGS: edit here only
# ============================================================

BASE_ROOT = Path("/home/yuebing/GenAI_density_urbanization/urban_data")
INFERENCE_ROOT = BASE_ROOT / "inference_outputs_multidim_ablation"
SWEEP_ROOT = INFERENCE_ROOT / "sweep"

EVAL_ROOT= Path("/home/yuebing/GenAI_density_urbanization/eval")
OUTPUT_DIR = EVAL_ROOT / "evaluation_clip_similarity"


EXPERIMENTS: Optional[List[str]] = ["baseline", "view_only", "view_level_only", "full"]
CKPT_TAGS: Optional[List[str]] = None
TARGET_DIMS: Optional[List[str]] = None
LEVELS = ["low", "medium", "high"]

COMMON_IMAGE_IDS_ONLY = False
FORCE_RECOMPUTE = False

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMAGE_SIZE = None
MAX_ROWS: Optional[int] = None
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

PAIR_KEY_COLS = ["experiment", "ckpt_tag", "image_id", "target_dim", "target_level", "generated_path"]
METADATA_USED_PATH = OUTPUT_DIR / "metadata_sweep_used.csv"
PAIRWISE_PATH = OUTPUT_DIR / "pairwise_clip_similarity_sweep.csv"
SUMMARY_BY_LEVEL_PATH = OUTPUT_DIR / "summary_clip_similarity_by_level.csv"
SUMMARY_BY_DIM_PATH = OUTPUT_DIR / "summary_clip_similarity_by_dim.csv"
SUMMARY_ACC_PATH = OUTPUT_DIR / "summary_clip_contrastive_level_accuracy.csv"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_level(x: str) -> str:
    s = str(x).strip().lower().replace("-", " ").replace("_", " ")
    aliases = {"lo": "low", "med": "medium", "mid": "medium", "hi": "high"}
    return aliases.get(s, s)


def level_to_tag(level: str) -> str:
    return normalize_level(level).replace(" ", "_")


def strip_sample_suffix(stem: str) -> str:
    return re.sub(r"_s\d+$", "", stem)


def file_signature(path: Path) -> str:
    try:
        st = path.stat()
        return f"{path.resolve()}|size={st.st_size}|mtime_ns={st.st_mtime_ns}"
    except Exception:
        return f"{path}|missing"


def text_hash(text: str) -> str:
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()


def make_key_series(df: pd.DataFrame, cols: Sequence[str] = PAIR_KEY_COLS) -> pd.Series:
    return df[list(cols)].astype(str).agg("||".join, axis=1)


def get_eval_config_hash() -> str:
    text = "|".join([
        f"clip_model={CLIP_MODEL_NAME}",
        f"levels={','.join(LEVELS)}",
        f"image_size={IMAGE_SIZE}",
        "target_text_version=v1",
        "contrastive_text=target_short",
    ])
    return hashlib.md5(text.encode("utf-8")).hexdigest()


EVAL_CONFIG_HASH = get_eval_config_hash()


# ============================================================
# 1. Build sweep metadata from folders
# ============================================================


def parse_level_from_filename(filename_stem: str, image_id: str, target_dim: str) -> Optional[str]:
    stem = strip_sample_suffix(filename_stem)
    prefix = f"{image_id}_{target_dim}_"
    if stem.startswith(prefix):
        return normalize_level(stem[len(prefix):])

    for lv in sorted(LEVELS, key=lambda x: len(level_to_tag(x)), reverse=True):
        tag = level_to_tag(lv)
        if stem.endswith("_" + tag):
            return normalize_level(lv)
    return None


def find_prompt_file(prompt_dir: Path, image_id: str, target_dim: str, target_level: str) -> Optional[Path]:
    if not prompt_dir.exists():
        return None

    tag = level_to_tag(target_level)
    p = prompt_dir / f"{image_id}_{target_dim}_{tag}.txt"
    if p.exists() and p.is_file():
        return p

    candidates = sorted(prompt_dir.glob(f"{image_id}_{target_dim}_*.txt"))
    for c in candidates:
        c_level = parse_level_from_filename(c.stem, image_id, target_dim)
        if c_level == normalize_level(target_level):
            return c
    return None


def build_sweep_metadata_from_files() -> pd.DataFrame:
    if not SWEEP_ROOT.exists():
        raise FileNotFoundError(f"Sweep root not found: {SWEEP_ROOT}")

    rows: List[Dict] = []
    valid_levels = [normalize_level(x) for x in LEVELS]

    for gen_path in SWEEP_ROOT.rglob("*"):
        if not gen_path.is_file():
            continue
        if gen_path.suffix.lower() not in IMAGE_EXTS:
            continue

        rel = gen_path.relative_to(SWEEP_ROOT)
        parts = rel.parts

        # Expected: <experiment>/<ckpt_tag>/<target_dim>/<image_id>/<file>.png
        if len(parts) < 5:
            continue

        experiment = parts[0]
        ckpt_tag = parts[1]
        target_dim = parts[2]
        image_id = parts[3]

        if target_dim == "prompts":
            continue
        if EXPERIMENTS is not None and experiment not in EXPERIMENTS:
            continue
        if CKPT_TAGS is not None and ckpt_tag not in CKPT_TAGS:
            continue
        if TARGET_DIMS is not None and target_dim not in TARGET_DIMS:
            continue

        target_level = parse_level_from_filename(gen_path.stem, image_id, target_dim)
        if target_level is None:
            print(f"[SKIP] cannot parse target level from: {gen_path}")
            continue
        target_level = normalize_level(target_level)
        if target_level not in valid_levels:
            continue

        prompt_dir = SWEEP_ROOT / experiment / ckpt_tag / "prompts"
        prompt_path = find_prompt_file(prompt_dir, image_id, target_dim, target_level)

        prompt = ""
        full_prompt_available = False
        if prompt_path is not None:
            try:
                prompt = prompt_path.read_text(encoding="utf-8")
                full_prompt_available = len(prompt.strip()) > 0
            except Exception:
                prompt = ""
                full_prompt_available = False

        rows.append({
            "mode": "sweep",
            "experiment": experiment,
            "ckpt_tag": ckpt_tag,
            "image_id": image_id,
            "target_dim": target_dim,
            "target_level": target_level,
            "generated_filename": gen_path.name,
            "generated_path": str(gen_path),
            "prompt_path": str(prompt_path) if prompt_path is not None else "",
            "prompt": prompt,
            "full_prompt_available": int(full_prompt_available),
            "prompt_hash": text_hash(prompt),
            "generated_file_signature": file_signature(gen_path),
            "eval_config_hash": EVAL_CONFIG_HASH,
        })

    meta = pd.DataFrame(rows)
    if meta.empty:
        raise RuntimeError(f"No valid sweep generated images found under {SWEEP_ROOT}")

    meta = meta.drop_duplicates(subset=PAIR_KEY_COLS).reset_index(drop=True)

    if COMMON_IMAGE_IDS_ONLY:
        meta = filter_common_image_ids(meta)

    if MAX_ROWS is not None and len(meta) > MAX_ROWS:
        meta = meta.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        print(f"[META] sampled MAX_ROWS={MAX_ROWS}")

    print("[PAIR BUILD] sweep rows built from folders")
    print(f"  total rows: {len(meta)}")
    print("  by experiment / ckpt:")
    print(meta.groupby(["experiment", "ckpt_tag"]).size().to_string())
    print("  by target_dim / target_level:")
    print(meta.groupby(["target_dim", "target_level"]).size().to_string())

    return meta


def filter_common_image_ids(meta: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment", "ckpt_tag", "target_dim", "target_level"]
    groups = list(meta.groupby(group_cols))
    if len(groups) <= 1:
        return meta.reset_index(drop=True)

    common_ids = None
    for _, g in groups:
        ids = set(g["image_id"].astype(str))
        common_ids = ids if common_ids is None else common_ids & ids

    common_ids = common_ids or set()
    before = len(meta)
    meta = meta[meta["image_id"].astype(str).isin(common_ids)].copy()
    print(f"[COMMON FILTER] common image_ids across {len(groups)} groups: {len(common_ids)}; rows {before} -> {len(meta)}")

    if meta.empty:
        raise RuntimeError(
            "COMMON_IMAGE_IDS_ONLY=True, but no common image_id exists across groups. "
            "Set COMMON_IMAGE_IDS_ONLY=False if you want to evaluate all available sweep images."
        )
    return meta.reset_index(drop=True)


# ============================================================
# 2. Text builders
# ============================================================


def build_target_short_text(target_dim: str, target_level: str) -> str:
    return f"a street view image with a {target_level} level of {target_dim} perception."


def build_target_static_text(target_dim: str, target_level: str) -> str:
    return (
        f"a static urban street view with a {target_level} level of {target_dim} perception, "
        "showing only urban form, road space, buildings, greenery, and streetscape features."
    )


# ============================================================
# 3. CLIP model and feature extraction
# ============================================================


def load_clip_model():
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as e:
        raise ImportError("Install dependencies with: pip install transformers") from e

    print(f"[CLIP] loading {CLIP_MODEL_NAME}")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
    return processor, model


def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if IMAGE_SIZE is not None:
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    return img


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def compute_text_features(texts: List[str], processor, model) -> torch.Tensor:
    text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
    with torch.no_grad():
        feats = l2_normalize(model.get_text_features(**text_inputs))
    return feats


def compute_image_features(images: List[Image.Image], processor, model) -> torch.Tensor:
    image_inputs = processor(images=images, return_tensors="pt")
    image_inputs = {k: v.to(DEVICE) for k, v in image_inputs.items()}
    with torch.no_grad():
        image_features = l2_normalize(model.get_image_features(**image_inputs))
    return image_features


def compute_image_and_text_sims(todo_meta: pd.DataFrame, processor, model) -> Tuple[pd.DataFrame, torch.Tensor]:
    rows = []
    all_image_features = []

    if todo_meta.empty:
        return pd.DataFrame(), torch.empty(0)

    print("[CLIP] computing full-prompt, target-short, target-static similarities...")
    for start in tqdm(range(0, len(todo_meta), BATCH_SIZE), desc="clip_batches"):
        batch = todo_meta.iloc[start:start + BATCH_SIZE].copy()
        images = [load_image(Path(p)) for p in batch["generated_path"].tolist()]

        short_texts = [build_target_short_text(str(r["target_dim"]), str(r["target_level"])) for _, r in batch.iterrows()]
        static_texts = [build_target_static_text(str(r["target_dim"]), str(r["target_level"])) for _, r in batch.iterrows()]

        full_texts = []
        full_available = []
        for _, r in batch.iterrows():
            p = str(r.get("prompt", "") or "").strip()
            ok = bool(int(r.get("full_prompt_available", 0))) and len(p) > 0
            full_available.append(ok)
            full_texts.append(p if ok else "a street view image")

        image_features = compute_image_features(images, processor, model)
        all_image_features.append(image_features.detach().cpu())

        full_features = compute_text_features(full_texts, processor, model)
        short_features = compute_text_features(short_texts, processor, model)
        static_features = compute_text_features(static_texts, processor, model)

        sim_full = (image_features * full_features).sum(dim=-1).detach().cpu().numpy()
        sim_short = (image_features * short_features).sum(dim=-1).detach().cpu().numpy()
        sim_static = (image_features * static_features).sum(dim=-1).detach().cpu().numpy()

        for i, (_, r) in enumerate(batch.iterrows()):
            out = r.to_dict()
            out["target_short_text"] = short_texts[i]
            out["target_static_text"] = static_texts[i]
            out["clip_sim_full_prompt"] = float(sim_full[i]) if full_available[i] else np.nan
            out["clip_sim_target_short"] = float(sim_short[i])
            out["clip_sim_target_static"] = float(sim_static[i])
            rows.append(out)

    image_features_cpu = torch.cat(all_image_features, dim=0)
    return pd.DataFrame(rows), image_features_cpu


def compute_text_features_for_candidates(processor, model, dims: List[str]) -> Dict[Tuple[str, str], torch.Tensor]:
    keys = []
    texts = []
    for dim in dims:
        for level in LEVELS:
            lv = normalize_level(level)
            keys.append((dim, lv))
            texts.append(build_target_short_text(dim, lv))

    feats = compute_text_features(texts, processor, model).detach().cpu()
    return {k: feats[i] for i, k in enumerate(keys)}


def add_contrastive_level_scores(df: pd.DataFrame, image_features: torch.Tensor, processor, model) -> pd.DataFrame:
    if df.empty:
        return df

    dims = sorted(df["target_dim"].astype(str).unique().tolist())
    candidate_feats = compute_text_features_for_candidates(processor, model, dims)
    levels_norm = [normalize_level(x) for x in LEVELS]

    rows = []
    for i, row in df.iterrows():
        img_feat = image_features[i]
        dim = str(row["target_dim"])
        scores = {}
        for lv in levels_norm:
            score = float((img_feat * candidate_feats[(dim, lv)]).sum().item())
            scores[f"clip_contrast_{lv}"] = score

        pred_level = max(levels_norm, key=lambda lv: scores[f"clip_contrast_{lv}"])
        sorted_scores = sorted([scores[f"clip_contrast_{lv}"] for lv in levels_norm], reverse=True)
        margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else np.nan

        out = row.to_dict()
        out.update(scores)
        out["clip_pred_level"] = pred_level
        out["clip_level_correct"] = int(pred_level == normalize_level(str(row["target_level"])))
        out["clip_contrast_margin_pred_minus_second"] = margin
        rows.append(out)

    return pd.DataFrame(rows)


# ============================================================
# 4. Incremental computation
# ============================================================

CORE_CLIP_COLS = [
    "clip_sim_target_short",
    "clip_sim_target_static",
    "clip_contrast_low",
    "clip_contrast_medium",
    "clip_contrast_high",
    "clip_pred_level",
    "clip_level_correct",
]


def split_clip_todo(meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if FORCE_RECOMPUTE or not PAIRWISE_PATH.exists():
        print("[INCREMENTAL] CLIP: recompute all or no previous pairwise file.")
        return pd.DataFrame(), meta.copy()

    old = pd.read_csv(PAIRWISE_PATH)
    required = PAIR_KEY_COLS + CORE_CLIP_COLS + ["eval_config_hash", "generated_file_signature", "prompt_hash"]
    if not all(c in old.columns for c in required):
        print("[INCREMENTAL] old CLIP pairwise file columns incomplete; recompute all.")
        return pd.DataFrame(), meta.copy()

    current_keys = set(make_key_series(meta))
    old = old.copy()
    old["_key"] = make_key_series(old)

    current_sig = meta[PAIR_KEY_COLS + ["generated_file_signature", "prompt_hash", "eval_config_hash"]].copy()
    current_sig["_key"] = make_key_series(current_sig)
    sig_map = {
        r["_key"]: (str(r["generated_file_signature"]), str(r["prompt_hash"]), str(r["eval_config_hash"]))
        for _, r in current_sig.iterrows()
    }

    keep_mask = []
    for _, r in old.iterrows():
        k = r["_key"]
        if k not in current_keys:
            keep_mask.append(False)
            continue
        sig = sig_map.get(k)
        if sig is None:
            keep_mask.append(False)
            continue
        keep_mask.append(
            str(r.get("generated_file_signature", "")) == sig[0]
            and str(r.get("prompt_hash", "")) == sig[1]
            and str(r.get("eval_config_hash", "")) == sig[2]
        )

    old_done = old[pd.Series(keep_mask, index=old.index)].copy()
    for c in CORE_CLIP_COLS:
        old_done = old_done[old_done[c].notna()].copy()

    done_keys = set(old_done["_key"])
    meta2 = meta.copy()
    meta2["_key"] = make_key_series(meta2)
    todo = meta2[~meta2["_key"].isin(done_keys)].copy()

    old_done = old_done.drop(columns=["_key"]).copy()
    todo = todo.drop(columns=["_key"]).copy()

    print(f"[INCREMENTAL] CLIP: existing valid rows={len(old_done)}, todo={len(todo)}, current total={len(meta)}")
    return old_done, todo


def compute_clip_for_todo(todo: pd.DataFrame) -> pd.DataFrame:
    if todo.empty:
        return pd.DataFrame()

    processor, model = load_clip_model()
    df, image_features = compute_image_and_text_sims(todo, processor, model)
    df = add_contrastive_level_scores(df, image_features, processor, model)
    return df


def build_pairwise_results(meta: pd.DataFrame) -> pd.DataFrame:
    old_done, todo = split_clip_todo(meta)

    if len(todo) > 0:
        new_df = compute_clip_for_todo(todo)
        pairwise = pd.concat([old_done, new_df], ignore_index=True)
    else:
        print("[INCREMENTAL] CLIP: no new rows to compute.")
        pairwise = old_done.copy()

    if pairwise.empty:
        raise RuntimeError("No CLIP pairwise results were produced.")

    current_keys = set(make_key_series(meta))
    pairwise["_key"] = make_key_series(pairwise)
    pairwise = pairwise[pairwise["_key"].isin(current_keys)].copy()
    pairwise = pairwise.drop_duplicates(subset=PAIR_KEY_COLS, keep="last")
    pairwise = pairwise.drop(columns=["_key"]).reset_index(drop=True)
    return pairwise


# ============================================================
# 5. Summaries
# ============================================================


def summarize_by_level(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment", "ckpt_tag", "target_dim", "target_level"]
    metrics = [
        "clip_sim_full_prompt",
        "clip_sim_target_short",
        "clip_sim_target_static",
        "clip_contrast_low",
        "clip_contrast_medium",
        "clip_contrast_high",
        "clip_level_correct",
        "clip_contrast_margin_pred_minus_second",
    ]
    metrics = [m for m in metrics if m in df.columns]

    rows = []
    for keys, g in df.groupby(group_cols):
        out = dict(zip(group_cols, keys))
        out["n"] = len(g)
        out["n_full_prompt_available"] = int(g["full_prompt_available"].fillna(0).astype(int).sum()) if "full_prompt_available" in g.columns else 0
        for m in metrics:
            out[f"{m}_mean"] = g[m].mean()
            out[f"{m}_std"] = g[m].std()
            out[f"{m}_median"] = g[m].median()
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_by_dim(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment", "ckpt_tag", "target_dim"]
    metrics = [
        "clip_sim_full_prompt",
        "clip_sim_target_short",
        "clip_sim_target_static",
        "clip_level_correct",
        "clip_contrast_margin_pred_minus_second",
    ]
    metrics = [m for m in metrics if m in df.columns]

    rows = []
    for keys, g in df.groupby(group_cols):
        out = dict(zip(group_cols, keys))
        out["n"] = len(g)
        out["n_full_prompt_available"] = int(g["full_prompt_available"].fillna(0).astype(int).sum()) if "full_prompt_available" in g.columns else 0
        for m in metrics:
            out[f"{m}_mean"] = g[m].mean()
            out[f"{m}_std"] = g[m].std()
            out[f"{m}_median"] = g[m].median()
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_contrastive_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (exp, ckpt), g in df.groupby(["experiment", "ckpt_tag"]):
        out = {
            "experiment": exp,
            "ckpt_tag": ckpt,
            "n": len(g),
            "clip_level_accuracy": g["clip_level_correct"].mean(),
            "clip_margin_mean": g["clip_contrast_margin_pred_minus_second"].mean(),
            "n_full_prompt_available": int(g["full_prompt_available"].fillna(0).astype(int).sum()) if "full_prompt_available" in g.columns else 0,
        }
        rows.append(out)
    return pd.DataFrame(rows)


# ============================================================
# 6. Main
# ============================================================


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    print(f"[INFO] DEVICE={DEVICE}")
    print(f"[INFO] INFERENCE_ROOT={INFERENCE_ROOT}")
    print(f"[INFO] SWEEP_ROOT={SWEEP_ROOT}")
    print(f"[INFO] OUTPUT_DIR={OUTPUT_DIR}")
    print(f"[INFO] FORCE_RECOMPUTE={FORCE_RECOMPUTE}")
    print(f"[INFO] EVAL_CONFIG_HASH={EVAL_CONFIG_HASH}")

    meta = build_sweep_metadata_from_files()
    meta.to_csv(METADATA_USED_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {METADATA_USED_PATH}")

    pairwise = build_pairwise_results(meta)
    pairwise.to_csv(PAIRWISE_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {PAIRWISE_PATH}")

    by_level = summarize_by_level(pairwise)
    by_level.to_csv(SUMMARY_BY_LEVEL_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {SUMMARY_BY_LEVEL_PATH}")

    by_dim = summarize_by_dim(pairwise)
    by_dim.to_csv(SUMMARY_BY_DIM_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {SUMMARY_BY_DIM_PATH}")

    acc = summarize_contrastive_accuracy(pairwise)
    acc.to_csv(SUMMARY_ACC_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {SUMMARY_ACC_PATH}")

    print("\n[DONE] CLIP similarity evaluation completed.")
    print(acc.sort_values(["experiment", "ckpt_tag"]).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("default")
    main()

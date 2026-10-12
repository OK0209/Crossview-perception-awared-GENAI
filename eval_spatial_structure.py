# -*- coding: utf-8 -*-
"""
Evaluate road-building spatial-structure consistency from folder structure, with incremental computation.

This script DOES NOT rely on metadata_all_both.csv or metadata_reconstruction.csv.
It builds pairs directly from:

    reconstruction/generated/<experiment>/<ckpt_tag>/<image_id>.png
    reconstruction/real/<image_id>.png

Main idea:
  Use semantic segmentation to extract ROAD and BUILDING masks from both generated and real images,
  then compare not only region overlap, but also road-building adjacency/contact and street-corridor profiles.

Recommended indicators for your paper:
  1) road_iou / building_iou: semantic region consistency
  2) road_building_contact_f1: road-building adjacency/contact consistency
  3) road_width_profile_mae: visible road corridor width profile error
  4) building_side_balance_abs_diff: left/right street-wall mass distribution difference
  5) road_to_building_setback_mean_abs_diff_px: road-building proximity/setback difference

Incremental behavior:
  - Already computed generated-real pairs are skipped.
  - Segmentation masks are cached by image path + file size + mtime + model/config signature.
  - Set FORCE_RECOMPUTE=True to recompute all spatial metrics.

Dependencies:
  pip install transformers accelerate pillow pandas tqdm opencv-python scikit-image
"""

from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


# ============================================================
# 0. USER SETTINGS: edit here only
# ============================================================

BASE_ROOT = Path("/home/yuebing/GenAI_density_urbanization/urban_data")

INFERENCE_ROOT = BASE_ROOT / "inference_outputs_untrained_baseline"
RECON_ROOT = INFERENCE_ROOT / "reconstruction"

REAL_DIR = RECON_ROOT / "real"
HINT_DIR = RECON_ROOT / "hint"
GENERATED_DIR = RECON_ROOT / "generated"

OUTPUT_DIR = Path("/home/yuebing/GenAI_density_urbanization/eval/evaluation_spatial_structure/untrained_baseline")

# 只评估这些 experiment；设为 None 表示扫描 generated 下所有 experiment
EXPERIMENTS: Optional[List[str]] =None # ["baseline", "view_only", "view_level_only", "full"]

# 只评估这些 ckpt_tag；设为 None 表示扫描所有 ckpt_tag
CKPT_TAGS: Optional[List[str]] = None
# CKPT_TAGS = ["epoch19-step135840", "epoch29-step203760"]

# 为了公平比较，是否只保留所有 experiment+ckpt 组共同拥有的 image_id
COMMON_IMAGE_IDS_ONLY = False

# 是否强制重算全部 pairwise spatial metrics
FORCE_RECOMPUTE = False

IMAGE_SIZE = 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Segmentation backend.
# Recommended formal setting: Mask2Former Cityscapes semantic.
# Faster fallback: set SEG_MODEL_TYPE="segformer" and use
# SEG_MODEL_NAME="nvidia/segformer-b0-finetuned-cityscapes-1024-1024".
SEG_MODEL_TYPE = "mask2former"
SEG_MODEL_NAME = "facebook/mask2former-swin-large-cityscapes-semantic"
SEG_CACHE_DIR = OUTPUT_DIR / "segmentation_cache"

# Contact boundary tolerance in pixels. Larger values tolerate small segmentation shifts.
CONTACT_TOLERANCE_PX = 5
BOUNDARY_KERNEL = 3

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ============================================================
# 1. Paths, keys, and config signatures
# ============================================================

PAIR_KEY_COLS = ["experiment", "ckpt_tag", "image_id", "generated_path", "real_path"]

METADATA_USED_PATH = OUTPUT_DIR / "metadata_reconstruction_used.csv"
PAIRWISE_PATH = OUTPUT_DIR / "pairwise_road_building_spatial_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary_road_building_spatial_metrics.csv"

CORE_METRIC_COLS = [
    "road_iou",
    "building_iou",
    "road_building_contact_f1",
    "road_width_profile_mae",
    "building_side_balance_abs_diff",
    "road_building_structure_score_0_1",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_metric_config_hash() -> str:
    text = "|".join(
        [
            f"seg_type={SEG_MODEL_TYPE}",
            f"seg_model={SEG_MODEL_NAME}",
            f"image_size={IMAGE_SIZE}",
            f"contact_tol={CONTACT_TOLERANCE_PX}",
            f"boundary_kernel={BOUNDARY_KERNEL}",
        ]
    )
    return hashlib.md5(text.encode("utf-8")).hexdigest()


METRIC_CONFIG_HASH = get_metric_config_hash()


def strip_sample_suffix(stem: str) -> str:
    """
    If NUM_SAMPLES > 1, generated files may be like:
        imageid_s00.png, imageid_s01.png
    The corresponding real file is usually:
        imageid.png
    """
    return re.sub(r"_s\d+$", "", stem)


def find_same_stem_file(folder: Path, stem: str, preferred_ext: Optional[str] = None) -> Optional[Path]:
    if preferred_ext:
        p = folder / f"{stem}{preferred_ext.lower()}"
        if p.exists() and p.is_file():
            return p

    for ext in sorted(IMAGE_EXTS):
        p = folder / f"{stem}{ext}"
        if p.exists() and p.is_file():
            return p
    return None


def make_key_series(df: pd.DataFrame, cols: Sequence[str] = PAIR_KEY_COLS) -> pd.Series:
    return df[list(cols)].astype(str).agg("||".join, axis=1)


def file_signature(path: Path) -> str:
    try:
        st = path.stat()
        return f"{path.resolve()}|size={st.st_size}|mtime_ns={st.st_mtime_ns}"
    except Exception:
        return f"{path}|missing"


# ============================================================
# 2. Build reconstruction pairs from folders
# ============================================================

def build_reconstruction_pairs_from_files() -> pd.DataFrame:
    """
    Build generated-real pairs from folder structure:

        GENERATED_DIR / <experiment> / <ckpt_tag> / <image_id>.png
        REAL_DIR / <image_id>.png

    This function does not use any CSV metadata.
    """
    if not GENERATED_DIR.exists():
        raise FileNotFoundError(f"Generated dir not found: {GENERATED_DIR}")
    if not REAL_DIR.exists():
        raise FileNotFoundError(f"Real dir not found: {REAL_DIR}")

    rows: List[Dict[str, str]] = []

    for gen_path in GENERATED_DIR.rglob("*"):
        if not gen_path.is_file():
            continue
        if gen_path.suffix.lower() not in IMAGE_EXTS:
            continue

        rel = gen_path.relative_to(GENERATED_DIR)
        parts = rel.parts

        # Expected: <experiment>/<ckpt_tag>/<image_file>
        if len(parts) < 3:
            print(f"[SKIP] unexpected generated path structure: {gen_path}")
            continue

        experiment = parts[0]
        ckpt_tag = parts[1]

        if EXPERIMENTS is not None and experiment not in EXPERIMENTS:
            continue
        if CKPT_TAGS is not None and ckpt_tag not in CKPT_TAGS:
            continue

        image_id = strip_sample_suffix(gen_path.stem)

        real_path = find_same_stem_file(REAL_DIR, image_id, preferred_ext=gen_path.suffix)
        if real_path is None:
            print(f"[MISSING REAL] image_id={image_id} for generated={gen_path}")
            continue

        hint_path = find_same_stem_file(HINT_DIR, image_id, preferred_ext=gen_path.suffix) if HINT_DIR.exists() else None

        rows.append(
            {
                "mode": "reconstruction",
                "experiment": experiment,
                "ckpt_tag": ckpt_tag,
                "image_id": image_id,
                "generated_filename": gen_path.name,
                "generated_path": str(gen_path),
                "real_path": str(real_path),
                "hint_path": str(hint_path) if hint_path is not None else "",
            }
        )

    meta = pd.DataFrame(rows)
    if meta.empty:
        raise RuntimeError(f"No valid generated-real pairs found under {GENERATED_DIR}")

    meta = meta.drop_duplicates(subset=PAIR_KEY_COLS).reset_index(drop=True)

    print("[PAIR BUILD] reconstruction pairs built from folders")
    print(f"  total pairs before common filtering: {len(meta)}")
    print("  by experiment / ckpt:")
    print(meta.groupby(["experiment", "ckpt_tag"]).size().to_string())

    return meta


def filter_common_image_ids(meta: pd.DataFrame) -> pd.DataFrame:
    if not COMMON_IMAGE_IDS_ONLY:
        return meta.reset_index(drop=True)

    groups = list(meta.groupby(["experiment", "ckpt_tag"]))
    if len(groups) <= 1:
        return meta.reset_index(drop=True)

    common_ids: Optional[Set[str]] = None
    for _, g in groups:
        ids = set(g["image_id"].astype(str))
        common_ids = ids if common_ids is None else common_ids & ids

    common_ids = common_ids or set()
    before = len(meta)
    meta = meta[meta["image_id"].astype(str).isin(common_ids)].copy()

    print(
        f"[COMMON FILTER] common image_ids across {len(groups)} groups: {len(common_ids)}; "
        f"rows {before} -> {len(meta)}"
    )

    if meta.empty:
        raise RuntimeError(
            "COMMON_IMAGE_IDS_ONLY=True, but no common image_id exists across experiment+ckpt groups. "
            "Set COMMON_IMAGE_IDS_ONLY=False if you want to evaluate all available images."
        )

    print("  by experiment / ckpt after common filtering:")
    print(meta.groupby(["experiment", "ckpt_tag"]).size().to_string())

    return meta.reset_index(drop=True)


# ============================================================
# 3. Segmentation model
# ============================================================

def load_segmentation_model():
    """
    Load a semantic segmentation model.

    Supported backends:
    - SEG_MODEL_TYPE = "mask2former"
      SEG_MODEL_NAME = "facebook/mask2former-swin-large-cityscapes-semantic"
      Uses Mask2FormerForUniversalSegmentation and
      processor.post_process_semantic_segmentation().

    - SEG_MODEL_TYPE = "segformer"
      SEG_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
      Uses SegformerForSemanticSegmentation and logits argmax.
    """
    try:
        from transformers import AutoImageProcessor
    except Exception as e:
        raise ImportError("Install dependencies with: pip install transformers accelerate") from e

    seg_type = SEG_MODEL_TYPE.lower().strip()
    print(f"[SEG] loading {seg_type}: {SEG_MODEL_NAME}")

    # use_fast=False avoids processor behavior changes across transformers versions.
    processor = AutoImageProcessor.from_pretrained(SEG_MODEL_NAME, use_fast=False)

    if seg_type == "mask2former":
        from transformers import Mask2FormerForUniversalSegmentation
        model = Mask2FormerForUniversalSegmentation.from_pretrained(SEG_MODEL_NAME).to(DEVICE).eval()
    elif seg_type == "segformer":
        from transformers import SegformerForSemanticSegmentation
        model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_NAME).to(DEVICE).eval()
    else:
        raise ValueError(f"Unsupported SEG_MODEL_TYPE={SEG_MODEL_TYPE!r}; use 'mask2former' or 'segformer'.")

    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}

    def find_label_ids(names: List[str]) -> List[int]:
        names_clean = {n.lower().replace("_", " ").replace("-", " ").strip() for n in names}
        out = []
        for i, label in id2label.items():
            label_clean = label.lower().replace("_", " ").replace("-", " ").strip()
            if label_clean in names_clean:
                out.append(i)
        return out

    road_ids = find_label_ids(["road"])
    building_ids = find_label_ids(["building"])
    sky_ids = find_label_ids(["sky"])
    vegetation_ids = find_label_ids(["vegetation"])

    if not road_ids or not building_ids:
        print("[WARN] Could not find exact road/building labels. Existing labels:")
        print(id2label)
        raise RuntimeError("Segmentation labels do not include road/building.")

    print(
        f"[SEG] road_ids={road_ids}, building_ids={building_ids}, "
        f"sky_ids={sky_ids}, vegetation_ids={vegetation_ids}"
    )

    return processor, model, road_ids, building_ids, sky_ids, vegetation_ids


def stable_cache_name(path: Path) -> str:
    """
    Cache key includes:
      - image path
      - file size and mtime
      - segmentation model / metric config hash
    This prevents stale cache if an image is overwritten at the same path.
    """
    text = f"{file_signature(path)}|seg_type={SEG_MODEL_TYPE}|seg_model={SEG_MODEL_NAME}|metric_config={METRIC_CONFIG_HASH}"
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:24]


def load_pil_rgb(path: Path, size: int = IMAGE_SIZE) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return img


def segment_one_image(
    path: Path,
    processor,
    model,
    road_ids: List[int],
    building_ids: List[int],
    sky_ids: Optional[List[int]] = None,
    vegetation_ids: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Segment one RGB image and return boolean masks.

    Important:
    Mask2Former outputs do not expose `outputs.logits` like SegFormer.
    For Mask2Former, use `processor.post_process_semantic_segmentation(...)`
    to obtain the final semantic label map.
    """
    ensure_dir(SEG_CACHE_DIR)
    cache_path = SEG_CACHE_DIR / f"{stable_cache_name(path)}_{IMAGE_SIZE}_{SEG_MODEL_TYPE.lower()}.npz"

    if cache_path.exists() and not FORCE_RECOMPUTE:
        data = np.load(cache_path)
        z = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        return {
            "road": data["road"].astype(bool),
            "building": data["building"].astype(bool),
            "sky": data["sky"].astype(bool) if "sky" in data.files else z,
            "vegetation": data["vegetation"].astype(bool) if "vegetation" in data.files else z,
            "label": data["label"],
        }

    img = load_pil_rgb(path)
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    seg_type = SEG_MODEL_TYPE.lower().strip()
    with torch.no_grad():
        outputs = model(**inputs)

        if seg_type == "mask2former":
            seg = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[(IMAGE_SIZE, IMAGE_SIZE)],
            )[0]
            pred = seg.detach().cpu().numpy().astype(np.int16)

        elif seg_type == "segformer":
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int16)

        else:
            raise ValueError(f"Unsupported SEG_MODEL_TYPE={SEG_MODEL_TYPE!r}")

    road = np.isin(pred, road_ids)
    building = np.isin(pred, building_ids)
    sky = np.isin(pred, sky_ids or [])
    vegetation = np.isin(pred, vegetation_ids or [])

    np.savez_compressed(
        cache_path,
        road=road.astype(np.uint8),
        building=building.astype(np.uint8),
        sky=sky.astype(np.uint8),
        vegetation=vegetation.astype(np.uint8),
        label=pred,
    )

    return {
        "road": road,
        "building": building,
        "sky": sky,
        "vegetation": vegetation,
        "label": pred,
    }


# ============================================================
# 4. Metrics
# ============================================================

def safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in [0, 0.0] else float("nan")


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return safe_div(inter, union)


def binary_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return safe_div(2 * inter, denom)


def morph_boundary(mask: np.ndarray, kernel_size: int = BOUNDARY_KERNEL) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dil = cv2.dilate(mask_u8, kernel, iterations=1)
    ero = cv2.erode(mask_u8, kernel, iterations=1)
    return (dil - ero).astype(bool)


def road_building_contact_map(
    road: np.ndarray,
    building: np.ndarray,
    tolerance_px: int = CONTACT_TOLERANCE_PX,
) -> np.ndarray:
    """
    Pixels near the adjacency/contact zone between road and building.
    """
    road = road.astype(bool)
    building = building.astype(bool)

    k = np.ones((max(1, tolerance_px), max(1, tolerance_px)), np.uint8)
    road_dil = cv2.dilate(road.astype(np.uint8), k, iterations=1).astype(bool)
    building_dil = cv2.dilate(building.astype(np.uint8), k, iterations=1).astype(bool)

    road_bd = morph_boundary(road)
    building_bd = morph_boundary(building)

    contact = (road_bd & building_dil) | (building_bd & road_dil)
    return contact.astype(bool)


def binary_f1(pred: np.ndarray, true: np.ndarray) -> float:
    pred = pred.astype(bool)
    true = true.astype(bool)
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    denom = 2 * tp + fp + fn
    return safe_div(2 * tp, denom)


def road_width_profile(road: np.ndarray, start_ratio: float = 0.45) -> np.ndarray:
    h = road.shape[0]
    y0 = int(h * start_ratio)
    return road[y0:, :].mean(axis=1).astype(np.float32)


def profile_mae(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(a[:n] - b[:n])))


def profile_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a = a[:n]
    b = b[:n]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def building_side_balance(building: np.ndarray, start_ratio: float = 0.25) -> float:
    """
    Left-right building mass balance in the street-view corridor region.
    Positive means more building mass on the left; negative means more on the right.
    """
    h, w = building.shape
    y0 = int(h * start_ratio)
    region = building[y0:, :]
    left = region[:, : w // 2].mean()
    right = region[:, w // 2 :].mean()
    return float(left - right)


def road_to_building_setback_mean(road: np.ndarray, building: np.ndarray) -> float:
    """
    Mean distance from road boundary pixels to the nearest building pixel, in pixels.
    """
    if road.sum() == 0 or building.sum() == 0:
        return float("nan")

    road_bd = morph_boundary(road)
    if road_bd.sum() == 0:
        return float("nan")

    # distanceTransform computes distance to zero pixels.
    # For building pixels as target zeros, use inverse building mask.
    inv_building = (~building.astype(bool)).astype(np.uint8)
    dist = cv2.distanceTransform(inv_building, cv2.DIST_L2, 5)
    vals = dist[road_bd]

    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def compute_pair_metrics(gen_masks: Dict[str, np.ndarray], real_masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    g_road, g_building = gen_masks["road"], gen_masks["building"]
    r_road, r_building = real_masks["road"], real_masks["building"]

    g_contact = road_building_contact_map(g_road, g_building)
    r_contact = road_building_contact_map(r_road, r_building)

    g_profile = road_width_profile(g_road)
    r_profile = road_width_profile(r_road)

    g_balance = building_side_balance(g_building)
    r_balance = building_side_balance(r_building)

    g_setback = road_to_building_setback_mean(g_road, g_building)
    r_setback = road_to_building_setback_mean(r_road, r_building)

    road_ratio_g = float(g_road.mean())
    road_ratio_r = float(r_road.mean())
    building_ratio_g = float(g_building.mean())
    building_ratio_r = float(r_building.mean())

    contact_len_g = int(g_contact.sum())
    contact_len_r = int(r_contact.sum())

    road_boundary_g = int(morph_boundary(g_road).sum())
    road_boundary_r = int(morph_boundary(r_road).sum())

    contact_ratio_g = safe_div(contact_len_g, road_boundary_g)
    contact_ratio_r = safe_div(contact_len_r, road_boundary_r)

    road_iou = binary_iou(g_road, r_road)
    building_iou = binary_iou(g_building, r_building)
    contact_f1 = binary_f1(g_contact, r_contact)
    road_prof_mae = profile_mae(g_profile, r_profile)
    side_diff = abs(g_balance - r_balance)

    # Composite score is optional. Use raw metrics as main evidence.
    components = [
        road_iou,
        building_iou,
        contact_f1,
        max(0.0, 1.0 - road_prof_mae),
        max(0.0, 1.0 - side_diff),
    ]
    components = [x for x in components if not np.isnan(x)]
    composite = float(np.mean(components)) if components else float("nan")

    return {
        "road_iou": road_iou,
        "road_dice": binary_dice(g_road, r_road),
        "building_iou": building_iou,
        "building_dice": binary_dice(g_building, r_building),
        "road_ratio_generated": road_ratio_g,
        "road_ratio_real": road_ratio_r,
        "road_ratio_abs_diff": abs(road_ratio_g - road_ratio_r),
        "building_ratio_generated": building_ratio_g,
        "building_ratio_real": building_ratio_r,
        "building_ratio_abs_diff": abs(building_ratio_g - building_ratio_r),
        "road_building_contact_f1": contact_f1,
        "contact_length_generated": contact_len_g,
        "contact_length_real": contact_len_r,
        "contact_length_abs_diff": abs(contact_len_g - contact_len_r),
        "contact_ratio_generated": contact_ratio_g,
        "contact_ratio_real": contact_ratio_r,
        "contact_ratio_abs_diff": abs(contact_ratio_g - contact_ratio_r)
        if not np.isnan(contact_ratio_g) and not np.isnan(contact_ratio_r)
        else np.nan,
        "road_width_profile_mae": road_prof_mae,
        "road_width_profile_corr": profile_corr(g_profile, r_profile),
        "building_side_balance_generated": g_balance,
        "building_side_balance_real": r_balance,
        "building_side_balance_abs_diff": side_diff,
        "road_to_building_setback_mean_generated_px": g_setback,
        "road_to_building_setback_mean_real_px": r_setback,
        "road_to_building_setback_mean_abs_diff_px": abs(g_setback - r_setback)
        if not np.isnan(g_setback) and not np.isnan(r_setback)
        else np.nan,
        "road_building_structure_score_0_1": composite,
    }


# ============================================================
# 5. Incremental pairwise spatial evaluation
# ============================================================

def split_spatial_todo(meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      old_done: already computed rows that still belong to current meta
      todo: current meta rows not yet computed under current metric config
    """
    if FORCE_RECOMPUTE or not PAIRWISE_PATH.exists():
        print("[INCREMENTAL] spatial metrics: recompute all or no previous pairwise file.")
        return pd.DataFrame(), meta.copy()

    old = pd.read_csv(PAIRWISE_PATH)

    required = PAIR_KEY_COLS + CORE_METRIC_COLS + ["metric_config_hash"]
    if not all(c in old.columns for c in required):
        print("[INCREMENTAL] old pairwise file columns incomplete or config hash missing; recompute all.")
        return pd.DataFrame(), meta.copy()

    current_keys = set(make_key_series(meta))

    old = old.copy()
    old["_key"] = make_key_series(old)

    old_done = old[
        old["_key"].isin(current_keys)
        & (old["metric_config_hash"].astype(str) == METRIC_CONFIG_HASH)
    ].copy()

    for c in CORE_METRIC_COLS:
        old_done = old_done[old_done[c].notna()].copy()

    done_keys = set(old_done["_key"])

    meta2 = meta.copy()
    meta2["_key"] = make_key_series(meta2)
    todo = meta2[~meta2["_key"].isin(done_keys)].copy()

    old_done = old_done.drop(columns=["_key"]).copy()
    todo = todo.drop(columns=["_key"]).copy()

    print(
        f"[INCREMENTAL] spatial metrics: existing valid rows={len(old_done)}, "
        f"todo={len(todo)}, current total={len(meta)}"
    )

    return old_done, todo


def evaluate_spatial_structure(todo_meta: pd.DataFrame) -> pd.DataFrame:
    if todo_meta.empty:
        return pd.DataFrame()

    processor, model, road_ids, building_ids, sky_ids, vegetation_ids = load_segmentation_model()
    rows: List[Dict] = []

    print("[EVAL] Computing road/building spatial metrics for todo rows...")
    for _, row in tqdm(todo_meta.iterrows(), total=len(todo_meta), desc="road_building"):
        out = row.to_dict()
        out["metric_config_hash"] = METRIC_CONFIG_HASH
        out["metric_error"] = ""

        gen_path = Path(row["generated_path"])
        real_path = Path(row["real_path"])

        try:
            gen_masks = segment_one_image(gen_path, processor, model, road_ids, building_ids, sky_ids, vegetation_ids)
            real_masks = segment_one_image(real_path, processor, model, road_ids, building_ids, sky_ids, vegetation_ids)
            metrics = compute_pair_metrics(gen_masks, real_masks)
            out.update(metrics)
        except Exception as e:
            out["metric_error"] = repr(e)
            print(f"[SKIP] {row.get('experiment')}/{row.get('ckpt_tag')}/{row.get('image_id')}: {e}")

        rows.append(out)

    return pd.DataFrame(rows)


def build_pairwise_results(meta: pd.DataFrame) -> pd.DataFrame:
    old_done, todo = split_spatial_todo(meta)

    if len(todo) > 0:
        new_pairwise = evaluate_spatial_structure(todo)
        pairwise = pd.concat([old_done, new_pairwise], ignore_index=True)
    else:
        print("[INCREMENTAL] spatial metrics: no new rows to compute.")
        pairwise = old_done.copy()

    if pairwise.empty:
        raise RuntimeError("No pairwise spatial results were produced.")

    current_keys = set(make_key_series(meta))
    pairwise["_key"] = make_key_series(pairwise)
    pairwise = pairwise[pairwise["_key"].isin(current_keys)].copy()
    pairwise = pairwise.drop_duplicates(subset=PAIR_KEY_COLS, keep="last")
    pairwise = pairwise.drop(columns=["_key"]).reset_index(drop=True)

    return pairwise


# ============================================================
# 6. Summary
# ============================================================

def summarize_metrics(pairwise: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "road_iou",
        "road_dice",
        "building_iou",
        "building_dice",
        "road_ratio_abs_diff",
        "building_ratio_abs_diff",
        "road_building_contact_f1",
        "contact_length_abs_diff",
        "contact_ratio_abs_diff",
        "road_width_profile_mae",
        "road_width_profile_corr",
        "building_side_balance_abs_diff",
        "road_to_building_setback_mean_abs_diff_px",
        "road_building_structure_score_0_1",
    ]
    metric_cols = [c for c in metric_cols if c in pairwise.columns]

    rows: List[Dict] = []
    for (exp, ckpt), g in pairwise.groupby(["experiment", "ckpt_tag"]):
        out = {
            "experiment": exp,
            "ckpt_tag": ckpt,
            "n_pairs": len(g),
        }
        for c in metric_cols:
            out[f"{c}_mean"] = g[c].mean()
            out[f"{c}_std"] = g[c].std()
            out[f"{c}_median"] = g[c].median()
        rows.append(out)

    return pd.DataFrame(rows)


# ============================================================
# 7. Main
# ============================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SEG_CACHE_DIR)

    print(f"[INFO] DEVICE={DEVICE}")
    print(f"[INFO] INFERENCE_ROOT={INFERENCE_ROOT}")
    print(f"[INFO] GENERATED_DIR={GENERATED_DIR}")
    print(f"[INFO] REAL_DIR={REAL_DIR}")
    print(f"[INFO] OUTPUT_DIR={OUTPUT_DIR}")
    print(f"[INFO] COMMON_IMAGE_IDS_ONLY={COMMON_IMAGE_IDS_ONLY}")
    print(f"[INFO] FORCE_RECOMPUTE={FORCE_RECOMPUTE}")
    print(f"[INFO] METRIC_CONFIG_HASH={METRIC_CONFIG_HASH}")

    meta = build_reconstruction_pairs_from_files()
    meta = filter_common_image_ids(meta)

    meta.to_csv(METADATA_USED_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {METADATA_USED_PATH}")

    pairwise = build_pairwise_results(meta)
    pairwise.to_csv(PAIRWISE_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {PAIRWISE_PATH}")

    summary = summarize_metrics(pairwise)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {SUMMARY_PATH}")

    print("\n[DONE] Road/building spatial-structure evaluation completed.")
    print(summary.sort_values(["experiment", "ckpt_tag"]).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("default")
    main()

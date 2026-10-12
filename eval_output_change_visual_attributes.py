# -*- coding: utf-8 -*-
"""
Output Change Analysis for sweep-generated images.

This standalone post-processing script reads already generated sweep outputs and
measures whether interpretable visual attributes change systematically when the
target perception level changes from low -> medium -> high.

It does not modify inference code, does not load Stable Diffusion / ControlNet,
and does not rerun sampling.

Dependencies:
    pip install transformers accelerate pillow pandas tqdm opencv-python scikit-image scipy matplotlib

Interpretation:
- enclosure_proxy: higher usually means stronger visual enclosure, not "better".
- vegetation_ratio: higher means more vegetation / greenery.
- edge_clutter_density: higher means denser edges; this can mean rich detail or clutter.
- line_orientation_consistency: higher means line directions are more concentrated, often more spatial order.
- texture_clutter: higher means more high-frequency texture / local variation.

These are output-change proxies, not perception scores. The goal is to test
systematic visual changes across low / medium / high, not to hard-code value
judgments.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# 0. USER CONFIGURATION
# ============================================================

BASE_ROOT = Path("/home/yuebing/GenAI_density_urbanization/urban_data")
INFERENCE_ROOT = BASE_ROOT / "inference_outputs_multidim_ablation"
SWEEP_ROOT = INFERENCE_ROOT / "sweep"

EVAL_ROOT = Path("/home/yuebing/GenAI_density_urbanization/eval")
OUTPUT_DIR = EVAL_ROOT / "output_change_visual_attributes"

EXPERIMENTS = ["baseline", "view_only", "view_level_only", "full"]
CKPT_TAGS = None
TARGET_DIMS = None
LEVELS = ["low", "medium", "high"]
FORCE_RECOMPUTE = False
COMMON_IMAGE_IDS_ONLY = False
IMAGE_SIZE = 512

# Supported:
#   SEG_BACKEND = "segformer"
#   SEG_BACKEND = "mask2former"
#
# Example Mask2Former Cityscapes checkpoint:
SEG_BACKEND = "mask2former"
SEG_MODEL_NAME = "facebook/mask2former-swin-small-cityscapes-semantic"
# SEG_BACKEND = "segformer"
# SEG_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
SEG_DEVICE = "auto"  # "auto" / "cuda" / "cpu"

CANNY_LOW = None
CANNY_HIGH = None
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 35
HOUGH_MAX_LINE_GAP = 8

MAKE_BOXPLOTS = True


# ============================================================
# 1. Constants / helpers
# ============================================================

SEG_CACHE_DIR = OUTPUT_DIR / "segmentation_cache"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS = [
    "enclosure_proxy",
    "vegetation_ratio",
    "edge_clutter_density",
    "line_orientation_consistency",
    "texture_clutter",
]
SEMANTIC_RATIO_METRICS = ["building_ratio", "sky_ratio", "road_ratio"]
LEVEL_ORDINAL = {"low": 0, "medium": 1, "high": 2}
SKIP_DIRS = {"prompts", "view_hint", "view_hints", "cross_attention", "control_analysis", "comparison", "real", "BEV"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_level(x: object) -> str:
    return str(x).strip().lower().replace("_", " ").replace("-", " ")


def file_signature(path: Path) -> str:
    try:
        stat = path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"
    except Exception:
        return "missing"


def metric_config_hash() -> str:
    payload = {
        "seg_backend": str(SEG_BACKEND),
        "seg_model": str(SEG_MODEL_NAME),
        "image_size": int(IMAGE_SIZE),
        "canny_low": CANNY_LOW,
        "canny_high": CANNY_HIGH,
        "hough_threshold": HOUGH_THRESHOLD,
        "hough_min_line_length": HOUGH_MIN_LINE_LENGTH,
        "hough_max_line_gap": HOUGH_MAX_LINE_GAP,
        "metrics": METRICS,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def cache_key_for_image(path: Path) -> str:
    payload = {
        "path": str(path.resolve()),
        "signature": file_signature(path),
        "seg_backend": str(SEG_BACKEND),
        "seg_model": str(SEG_MODEL_NAME),
        "image_size": int(IMAGE_SIZE),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def read_rgb(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img)


def read_gray(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    rgb = read_rgb(path, size)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def safe_spearman(x: List[float], y: List[float]) -> float:
    valid = [(a, b) for a, b in zip(x, y) if pd.notna(a) and pd.notna(b)]
    if len(valid) < 3:
        return np.nan
    xv = np.asarray([a for a, _ in valid], dtype=np.float64)
    yv = np.asarray([b for _, b in valid], dtype=np.float64)
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(xv, yv).correlation)
    except Exception:
        xr = pd.Series(xv).rank(method="average").to_numpy()
        yr = pd.Series(yv).rank(method="average").to_numpy()
        if xr.std() <= 1e-12 or yr.std() <= 1e-12:
            return np.nan
        return float(np.corrcoef(xr, yr)[0, 1])


# ============================================================
# 2. Sweep image scanning
# ============================================================

def parse_generated_image_path(path: Path) -> Optional[Dict]:
    try:
        rel = path.relative_to(SWEEP_ROOT)
        parts = rel.parts
        if len(parts) < 5:
            return None
        experiment, ckpt_tag, target_dim, image_id = parts[0], parts[1], parts[2], parts[3]
        if experiment in SKIP_DIRS or target_dim in SKIP_DIRS:
            return None
        if EXPERIMENTS and experiment not in EXPERIMENTS:
            return None
        if CKPT_TAGS is not None and ckpt_tag not in CKPT_TAGS:
            return None
        if TARGET_DIMS is not None and target_dim not in TARGET_DIMS:
            return None
        if path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
            return None

        stem = path.stem
        prefix = f"{image_id}_{target_dim}_"
        if stem.startswith(prefix):
            level = normalize_level(stem[len(prefix):])
        else:
            toks = stem.split("_")
            if not toks:
                return None
            level = normalize_level(toks[-1])
        if level not in LEVELS:
            return None

        return {
            "experiment": experiment,
            "ckpt_tag": ckpt_tag,
            "image_id": image_id,
            "target_dim": target_dim,
            "target_level": level,
            "generated_path": str(path),
            "generated_filename": path.name,
            "generated_file_signature": file_signature(path),
        }
    except Exception:
        return None


def build_sweep_metadata_from_files() -> pd.DataFrame:
    rows = []
    warnings_list = []
    if not SWEEP_ROOT.exists():
        print(f"[WARN] SWEEP_ROOT does not exist: {SWEEP_ROOT}")
        return pd.DataFrame()

    for exp_dir in SWEEP_ROOT.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in SKIP_DIRS:
            continue
        if EXPERIMENTS and exp_dir.name not in EXPERIMENTS:
            continue
        for ckpt_dir in exp_dir.iterdir():
            if not ckpt_dir.is_dir() or ckpt_dir.name in SKIP_DIRS:
                continue
            if CKPT_TAGS is not None and ckpt_dir.name not in CKPT_TAGS:
                continue
            for p in ckpt_dir.rglob("*"):
                if not p.is_file():
                    continue
                if any(part in SKIP_DIRS for part in p.relative_to(ckpt_dir).parts[:-1]):
                    continue
                parsed = parse_generated_image_path(p)
                if parsed is None:
                    if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                        warnings_list.append(str(p))
                    continue
                rows.append(parsed)

    df = pd.DataFrame(rows).drop_duplicates()
    if not df.empty:
        order = {x: i for i, x in enumerate(LEVELS)}
        df["_level_order"] = df["target_level"].map(order)
        df = df.sort_values(["experiment", "ckpt_tag", "target_dim", "image_id", "_level_order"]).drop(columns=["_level_order"])

    if COMMON_IMAGE_IDS_ONLY and not df.empty:
        group_cols = ["experiment", "ckpt_tag", "target_dim", "image_id"]
        complete = df.groupby(group_cols)["target_level"].nunique().reset_index()
        complete = complete[complete["target_level"] >= len(LEVELS)][group_cols]
        df = df.merge(complete, on=group_cols, how="inner")

    ensure_dir(OUTPUT_DIR)
    if warnings_list:
        (OUTPUT_DIR / "unparsed_generated_files.txt").write_text("\n".join(warnings_list), encoding="utf-8")
    df.to_csv(OUTPUT_DIR / "metadata_sweep_used.csv", index=False, encoding="utf-8-sig")
    return df


# ============================================================
# 3. Segmentation
# ============================================================

@dataclass
class SegmentationBundle:
    processor: object
    model: object
    device: str
    label_ids: Dict[str, List[int]]
    backend: str


def _ids_for_labels(id2label: Dict, names: List[str]) -> List[int]:
    out = []
    wanted = {n.lower() for n in names}
    for k, v in id2label.items():
        label = str(v).lower().replace("-", " ").replace("_", " ")
        if label in wanted:
            out.append(int(k))
    return out


def load_segmentation_model() -> SegmentationBundle:
    import torch
    from transformers import AutoImageProcessor

    device = "cuda" if SEG_DEVICE == "auto" and torch.cuda.is_available() else SEG_DEVICE
    if device == "auto":
        device = "cpu"

    backend = str(SEG_BACKEND).strip().lower()
    processor = AutoImageProcessor.from_pretrained(SEG_MODEL_NAME)
    if backend == "segformer":
        from transformers import SegformerForSemanticSegmentation
        model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_NAME)
    elif backend == "mask2former":
        from transformers import Mask2FormerForUniversalSegmentation
        model = Mask2FormerForUniversalSegmentation.from_pretrained(SEG_MODEL_NAME)
    else:
        raise ValueError("SEG_BACKEND must be 'segformer' or 'mask2former'.")

    model.to(device)
    model.eval()

    id2label = getattr(model.config, "id2label", {})
    label_ids = {
        "vegetation": _ids_for_labels(id2label, ["vegetation"]),
        "building": _ids_for_labels(id2label, ["building"]),
        "sky": _ids_for_labels(id2label, ["sky"]),
        "road": _ids_for_labels(id2label, ["road"]),
    }
    # Cityscapes fallback ids for SegFormer/Mask2Former checkpoints.
    label_ids.setdefault("vegetation", [])
    if not label_ids["vegetation"]:
        label_ids["vegetation"] = [8]
    if not label_ids["building"]:
        label_ids["building"] = [2]
    if not label_ids["sky"]:
        label_ids["sky"] = [10]
    if not label_ids["road"]:
        label_ids["road"] = [0]

    print(f"[SEG] backend={backend}, model={SEG_MODEL_NAME}, device={device}")
    print(f"[SEG] label ids: {label_ids}")
    return SegmentationBundle(processor=processor, model=model, device=device, label_ids=label_ids, backend=backend)


def semantic_label_map_from_outputs(outputs, seg: SegmentationBundle) -> np.ndarray:
    import torch

    if seg.backend == "segformer":
        logits = outputs.logits
        logits = torch.nn.functional.interpolate(
            logits,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        return logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    if seg.backend == "mask2former":
        label = seg.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(IMAGE_SIZE, IMAGE_SIZE)],
        )[0]
        return label.detach().cpu().numpy().astype(np.uint8)

    raise ValueError(f"Unsupported segmentation backend: {seg.backend}")


def segment_one_image_with_cache(image_path: Path, seg: SegmentationBundle) -> Dict[str, np.ndarray]:
    import torch

    ensure_dir(SEG_CACHE_DIR)
    key = cache_key_for_image(image_path)
    cache_path = SEG_CACHE_DIR / f"{key}.npz"
    if cache_path.exists() and not FORCE_RECOMPUTE:
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}

    img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    inputs = seg.processor(images=img, return_tensors="pt")
    inputs = {k: v.to(seg.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = seg.model(**inputs)
        label_map = semantic_label_map_from_outputs(outputs, seg)

    masks = {"label_map": label_map}
    for name, ids in seg.label_ids.items():
        masks[f"{name}_mask"] = np.isin(label_map, np.asarray(ids, dtype=np.uint8))
    np.savez_compressed(cache_path, **masks)
    return masks


def compute_semantic_ratios(seg_maps: Dict[str, np.ndarray]) -> Dict[str, float]:
    total = int(seg_maps["label_map"].size)
    vegetation_pixels = int(seg_maps["vegetation_mask"].sum())
    building_pixels = int(seg_maps["building_mask"].sum())
    sky_pixels = int(seg_maps["sky_mask"].sum())
    road_pixels = int(seg_maps["road_mask"].sum())
    vegetation_ratio = vegetation_pixels / max(total, 1)
    building_ratio = building_pixels / max(total, 1)
    sky_ratio = sky_pixels / max(total, 1)
    road_ratio = road_pixels / max(total, 1)
    enclosure_proxy = building_ratio / (sky_ratio + road_ratio + 1e-6)
    return {
        "vegetation_ratio": vegetation_ratio,
        "vegetation_pixels": vegetation_pixels,
        "total_pixels": total,
        "building_ratio": building_ratio,
        "sky_ratio": sky_ratio,
        "road_ratio": road_ratio,
        "enclosure_proxy": enclosure_proxy,
    }


# ============================================================
# 4. Visual metrics
# ============================================================

def compute_edge_clutter_density(gray: np.ndarray) -> Dict[str, float]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if CANNY_LOW is None or CANNY_HIGH is None:
        med = float(np.median(blur))
        low = int(max(0, 0.66 * med))
        high = int(min(255, 1.33 * med))
    else:
        low, high = int(CANNY_LOW), int(CANNY_HIGH)
    edges = cv2.Canny(blur, low, high)
    edge_pixels = int((edges > 0).sum())
    total = int(edges.size)
    return {
        "edge_clutter_density": edge_pixels / max(total, 1),
        "edge_pixels": edge_pixels,
        "canny_low": low,
        "canny_high": high,
        "edge_map": edges,
    }


def compute_line_orientation_consistency(edge_map: np.ndarray) -> Dict[str, float]:
    lines = cv2.HoughLinesP(
        edge_map,
        rho=1,
        theta=np.pi / 180.0,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    if lines is None or len(lines) == 0:
        return {
            "line_orientation_consistency": np.nan,
            "line_count": 0,
            "line_total_length": 0.0,
            "dominant_orientation_deg": np.nan,
        }

    theta_vals = []
    weights = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length <= 1e-6:
            continue
        theta = math.atan2(dy, dx) % math.pi
        theta_vals.append(theta)
        weights.append(length)

    if not theta_vals:
        return {
            "line_orientation_consistency": np.nan,
            "line_count": 0,
            "line_total_length": 0.0,
            "dominant_orientation_deg": np.nan,
        }

    theta = np.asarray(theta_vals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    c = np.sum(w * np.cos(2.0 * theta))
    s = np.sum(w * np.sin(2.0 * theta))
    denom = np.sum(w)
    R = float(np.sqrt(c * c + s * s) / denom) if denom > 1e-12 else np.nan
    dominant = 0.5 * math.atan2(s, c)
    dominant_deg = (math.degrees(dominant) % 180.0)
    return {
        "line_orientation_consistency": R,
        "line_count": int(len(theta_vals)),
        "line_total_length": float(denom),
        "dominant_orientation_deg": float(dominant_deg),
    }


def compute_texture_clutter(gray: np.ndarray) -> Dict[str, float]:
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    lap_std = float(np.std(lap))
    lap_mean_abs = float(np.mean(np.abs(lap)))
    # Higher std(Laplacian) means stronger high-frequency texture/local variation.
    return {
        "texture_clutter": lap_std / 255.0,
        "laplacian_std": lap_std,
        "laplacian_mean_abs": lap_mean_abs,
    }


def compute_one_image_metrics(row: pd.Series, seg: SegmentationBundle, cfg_hash: str) -> Dict:
    path = Path(str(row["generated_path"]))
    out = row.to_dict()
    out["metric_config_hash"] = cfg_hash
    out["metric_error"] = ""
    try:
        seg_maps = segment_one_image_with_cache(path, seg)
        out.update(compute_semantic_ratios(seg_maps))

        gray = read_gray(path, IMAGE_SIZE)
        edge = compute_edge_clutter_density(gray)
        edge_map = edge.pop("edge_map")
        out.update(edge)
        out.update(compute_line_orientation_consistency(edge_map))
        out.update(compute_texture_clutter(gray))
    except Exception as e:
        out["metric_error"] = str(e)
        for col in METRICS + SEMANTIC_RATIO_METRICS:
            out.setdefault(col, np.nan)
    return out


# ============================================================
# 5. Incremental computation
# ============================================================

def split_todo_by_existing(meta: pd.DataFrame, cfg_hash: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pair_path = OUTPUT_DIR / "pairwise_output_change_visual_metrics.csv"
    if FORCE_RECOMPUTE or not pair_path.exists():
        return meta, pd.DataFrame()
    existing = pd.read_csv(pair_path)
    if existing.empty:
        return meta, existing

    key_cols = ["experiment", "ckpt_tag", "image_id", "target_dim", "target_level", "generated_path"]
    metric_ok = existing[METRICS].notna().all(axis=1) if set(METRICS).issubset(existing.columns) else False
    reusable = existing[
        (existing.get("metric_config_hash", "") == cfg_hash)
        & metric_ok
    ].copy()
    if reusable.empty:
        return meta, existing

    reusable_keys = set(tuple(x) for x in reusable[key_cols].astype(str).to_numpy())
    sig_map = {
        tuple(r[key_cols].astype(str)): str(r.get("generated_file_signature", ""))
        for _, r in reusable.iterrows()
    }
    keep_todo = []
    for _, r in meta.iterrows():
        key = tuple(str(r[c]) for c in key_cols)
        if key in reusable_keys and sig_map.get(key) == str(r["generated_file_signature"]):
            continue
        keep_todo.append(r)
    todo = pd.DataFrame(keep_todo)
    return todo, existing


def build_pairwise_results(meta: pd.DataFrame) -> pd.DataFrame:
    ensure_dir(OUTPUT_DIR)
    cfg_hash = metric_config_hash()
    todo, existing = split_todo_by_existing(meta, cfg_hash)
    print(f"[INFO] scanned images: {len(meta)} | todo: {len(todo)} | existing file rows: {len(existing)}")

    new_rows = []
    if not todo.empty:
        seg = load_segmentation_model()
        for _, row in tqdm(todo.iterrows(), total=len(todo), desc="metrics"):
            new_rows.append(compute_one_image_metrics(row, seg, cfg_hash))

    new_df = pd.DataFrame(new_rows)
    if FORCE_RECOMPUTE or existing.empty:
        out = new_df
    else:
        key_cols = ["experiment", "ckpt_tag", "image_id", "target_dim", "target_level", "generated_path"]
        if not new_df.empty:
            new_keys = set(tuple(x) for x in new_df[key_cols].astype(str).to_numpy())
            existing = existing[
                ~existing[key_cols].astype(str).apply(lambda r: tuple(r), axis=1).isin(new_keys)
            ]
        out = pd.concat([existing, new_df], ignore_index=True, sort=False)

    if not out.empty:
        out = out.drop_duplicates(
            subset=["experiment", "ckpt_tag", "image_id", "target_dim", "target_level", "generated_path"],
            keep="last",
        )
    out.to_csv(OUTPUT_DIR / "pairwise_output_change_visual_metrics.csv", index=False, encoding="utf-8-sig")
    return out


# ============================================================
# 6. Summaries
# ============================================================

def summarize_by_level(pairwise: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment", "ckpt_tag", "target_dim", "target_level"]
    value_cols = METRICS + SEMANTIC_RATIO_METRICS
    rows = []
    for keys, g in pairwise.groupby(group_cols):
        rec = dict(zip(group_cols, keys))
        rec["n"] = int(len(g))
        for col in value_cols:
            vals = pd.to_numeric(g[col], errors="coerce") if col in g.columns else pd.Series(dtype=float)
            rec[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            rec[f"{col}_std"] = float(vals.std()) if vals.notna().sum() > 1 else np.nan
            rec[f"{col}_median"] = float(vals.median()) if vals.notna().any() else np.nan
            rec[f"{col}_valid_n"] = int(vals.notna().sum())
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "summary_visual_attributes_by_level.csv", index=False, encoding="utf-8-sig")
    return out


def summarize_by_dim(pairwise: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment", "ckpt_tag", "target_dim"]
    value_cols = METRICS + SEMANTIC_RATIO_METRICS
    rows = []
    for keys, g in pairwise.groupby(group_cols):
        rec = dict(zip(group_cols, keys))
        rec["n"] = int(len(g))
        for col in value_cols:
            vals = pd.to_numeric(g[col], errors="coerce") if col in g.columns else pd.Series(dtype=float)
            rec[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            rec[f"{col}_std"] = float(vals.std()) if vals.notna().sum() > 1 else np.nan
            rec[f"{col}_median"] = float(vals.median()) if vals.notna().any() else np.nan
            rec[f"{col}_valid_n"] = int(vals.notna().sum())
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "summary_visual_attributes_by_dim.csv", index=False, encoding="utf-8-sig")
    return out


def build_low_medium_high_wide(pairwise: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["experiment", "ckpt_tag", "image_id", "target_dim"]
    value_cols = METRICS + SEMANTIC_RATIO_METRICS
    rows = []
    for keys, g in pairwise.groupby(key_cols):
        rec = dict(zip(key_cols, keys))
        for metric in value_cols:
            for level in LEVELS:
                vals = g[g["target_level"] == level][metric] if metric in g.columns else pd.Series(dtype=float)
                rec[f"{metric}_{level}"] = float(pd.to_numeric(vals, errors="coerce").mean()) if len(vals) else np.nan
            rec[f"{metric}_high_minus_low"] = rec.get(f"{metric}_high", np.nan) - rec.get(f"{metric}_low", np.nan)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "output_change_low_medium_high_wide.csv", index=False, encoding="utf-8-sig")
    return out


def summarize_output_change_trends(wide: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment", "ckpt_tag", "target_dim"]
    rows = []
    for keys, g in wide.groupby(group_cols):
        rec_base = dict(zip(group_cols, keys))
        for metric in METRICS:
            rec = dict(rec_base)
            rec["metric"] = metric
            delta = pd.to_numeric(g.get(f"{metric}_high_minus_low"), errors="coerce")
            rec["high_minus_low_mean"] = float(delta.mean()) if delta.notna().any() else np.nan
            rec["high_minus_low_std"] = float(delta.std()) if delta.notna().sum() > 1 else np.nan
            rec["high_minus_low_median"] = float(delta.median()) if delta.notna().any() else np.nan

            cors = []
            inc = 0
            dec = 0
            valid_triplets = 0
            for _, r in g.iterrows():
                ys = [r.get(f"{metric}_{lv}", np.nan) for lv in LEVELS]
                if all(pd.notna(y) for y in ys):
                    valid_triplets += 1
                    inc += int(ys[0] <= ys[1] <= ys[2])
                    dec += int(ys[0] >= ys[1] >= ys[2])
                    cors.append(safe_spearman([LEVEL_ORDINAL[lv] for lv in LEVELS], ys))
            cors = pd.Series(cors, dtype=float)
            rec["spearman_level_metric_mean"] = float(cors.mean()) if cors.notna().any() else np.nan
            rec["monotonic_increase_rate"] = inc / valid_triplets if valid_triplets else np.nan
            rec["monotonic_decrease_rate"] = dec / valid_triplets if valid_triplets else np.nan
            rec["valid_triplet_n"] = int(valid_triplets)
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "summary_output_change_trends.csv", index=False, encoding="utf-8-sig")
    return out


# ============================================================
# 7. Figures
# ============================================================

def make_level_trend_figures(summary_level: pd.DataFrame) -> None:
    ensure_dir(FIGURE_DIR / "level_trends")
    if summary_level.empty:
        return
    colors = {exp: c for exp, c in zip(EXPERIMENTS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])}
    for ckpt_tag in sorted(summary_level["ckpt_tag"].astype(str).unique()):
        ck = summary_level[summary_level["ckpt_tag"].astype(str) == ckpt_tag]
        for target_dim in sorted(ck["target_dim"].astype(str).unique()):
            sub = ck[ck["target_dim"].astype(str) == target_dim]
            for metric in METRICS:
                plt.figure(figsize=(6, 4))
                for exp in EXPERIMENTS:
                    g = sub[sub["experiment"].astype(str) == exp].copy()
                    if g.empty:
                        continue
                    xs, ys = [], []
                    for lv in LEVELS:
                        row = g[g["target_level"].astype(str) == lv]
                        if len(row):
                            xs.append(LEVEL_ORDINAL[lv])
                            ys.append(float(row.iloc[0].get(f"{metric}_mean", np.nan)))
                    if xs:
                        plt.plot(xs, ys, marker="o", label=exp, color=colors.get(exp))
                plt.xticks([0, 1, 2], LEVELS)
                plt.xlabel("target level")
                plt.ylabel(metric)
                plt.title(f"{ckpt_tag} | {target_dim} | {metric}")
                plt.legend()
                plt.tight_layout()
                out = FIGURE_DIR / "level_trends" / f"trend__{ckpt_tag}__{target_dim}__{metric}.png"
                plt.savefig(out, dpi=160)
                plt.close()


def make_delta_heatmaps(trends: pd.DataFrame) -> None:
    ensure_dir(FIGURE_DIR / "high_minus_low_heatmaps")
    if trends.empty:
        return
    for ckpt_tag in sorted(trends["ckpt_tag"].astype(str).unique()):
        ck = trends[trends["ckpt_tag"].astype(str) == ckpt_tag]
        for target_dim in sorted(ck["target_dim"].astype(str).unique()):
            sub = ck[ck["target_dim"].astype(str) == target_dim]
            mat = np.full((len(EXPERIMENTS), len(METRICS)), np.nan, dtype=float)
            for i, exp in enumerate(EXPERIMENTS):
                for j, metric in enumerate(METRICS):
                    row = sub[(sub["experiment"].astype(str) == exp) & (sub["metric"].astype(str) == metric)]
                    if len(row):
                        mat[i, j] = float(row.iloc[0].get("high_minus_low_mean", np.nan))
            plt.figure(figsize=(9, 4))
            im = plt.imshow(mat, aspect="auto", cmap="coolwarm")
            plt.colorbar(im, label="high - low mean")
            plt.yticks(range(len(EXPERIMENTS)), EXPERIMENTS)
            plt.xticks(range(len(METRICS)), METRICS, rotation=35, ha="right")
            plt.title(f"{ckpt_tag} | {target_dim} | high-low deltas")
            plt.tight_layout()
            out = FIGURE_DIR / "high_minus_low_heatmaps" / f"delta_heatmap__{ckpt_tag}__{target_dim}.png"
            plt.savefig(out, dpi=160)
            plt.close()


def make_boxplots(pairwise: pd.DataFrame) -> None:
    if not MAKE_BOXPLOTS or pairwise.empty:
        return
    ensure_dir(FIGURE_DIR / "boxplots")
    for target_dim in sorted(pairwise["target_dim"].astype(str).unique()):
        sub = pairwise[pairwise["target_dim"].astype(str) == target_dim]
        for metric in METRICS:
            plt.figure(figsize=(10, 5))
            positions = []
            data = []
            labels = []
            pos = 1
            for exp in EXPERIMENTS:
                for lv in LEVELS:
                    vals = pd.to_numeric(
                        sub[(sub["experiment"].astype(str) == exp) & (sub["target_level"].astype(str) == lv)][metric],
                        errors="coerce",
                    ).dropna().to_numpy()
                    data.append(vals)
                    positions.append(pos)
                    labels.append(f"{exp}\n{lv}")
                    pos += 1
                pos += 1
            plt.boxplot(data, positions=positions, showfliers=False)
            plt.xticks(positions, labels, rotation=45, ha="right", fontsize=8)
            plt.ylabel(metric)
            plt.title(f"{target_dim} | {metric}")
            plt.tight_layout()
            out = FIGURE_DIR / "boxplots" / f"boxplot__{target_dim}__{metric}.png"
            plt.savefig(out, dpi=160)
            plt.close()


def make_figures(pairwise: pd.DataFrame, summary_level: pd.DataFrame, trends: pd.DataFrame) -> None:
    make_level_trend_figures(summary_level)
    make_delta_heatmaps(trends)
    make_boxplots(pairwise)


def write_readme() -> None:
    text = """# Output Change Visual Attributes

This analysis measures interpretable output-change proxies from sweep-generated images.
It does not estimate perception scores directly.

Metrics:
- enclosure_proxy = building_ratio / (sky_ratio + road_ratio + epsilon). Higher usually means stronger enclosure.
- vegetation_ratio = vegetation pixels / total pixels. Higher means more greenery.
- edge_clutter_density = Canny edge pixels / total pixels. Higher means denser edges, which can indicate detail or clutter.
- line_orientation_consistency = weighted circular resultant length of line orientations. Higher means more concentrated line directions.
- texture_clutter = std(Laplacian(gray)) / 255. Higher means stronger high-frequency texture/local variation.

Interpretation should be dimension-specific and empirical. Do not assume high is always good.

Example hypotheses:
- beautiful high may increase vegetation_ratio, reduce or moderate edge_clutter_density, or improve line_orientation_consistency.
- safety high may improve line_orientation_consistency and reduce edge_clutter_density.
- lively high may increase vegetation_ratio, edge_clutter_density, or texture_clutter.
- boring high may reduce texture_clutter, edge_clutter_density, or vegetation_ratio.
- depressing high may reduce vegetation_ratio or change enclosure/texture.

Final conclusions should come from the statistics, not from these hypotheses.
"""
    (OUTPUT_DIR / "README.md").write_text(text, encoding="utf-8")


# ============================================================
# 8. Main
# ============================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SEG_CACHE_DIR)
    ensure_dir(FIGURE_DIR)
    write_readme()

    meta = build_sweep_metadata_from_files()
    if meta.empty:
        print("[WARN] No generated sweep images found.")
        return

    pairwise = build_pairwise_results(meta)
    if pairwise.empty:
        print("[WARN] No pairwise metrics were produced.")
        return

    summary_level = summarize_by_level(pairwise)
    summarize_by_dim(pairwise)
    wide = build_low_medium_high_wide(pairwise)
    trends = summarize_output_change_trends(wide)
    make_figures(pairwise, summary_level, trends)

    print("\n========== Output Change Analysis ==========")
    counts = pairwise.groupby(["experiment", "ckpt_tag", "target_dim"]).size().reset_index(name="n")
    print(counts.to_string(index=False))
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Generated:")
    for name in [
        "metadata_sweep_used.csv",
        "pairwise_output_change_visual_metrics.csv",
        "summary_visual_attributes_by_level.csv",
        "summary_visual_attributes_by_dim.csv",
        "output_change_low_medium_high_wide.csv",
        "summary_output_change_trends.csv",
        "README.md",
    ]:
        print(f"  - {OUTPUT_DIR / name}")
    print(f"  - figures: {FIGURE_DIR}")
    print("===========================================\n")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

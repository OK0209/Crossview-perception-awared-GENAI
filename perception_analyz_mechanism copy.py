# -*- coding: utf-8 -*-
"""
Post-hoc analysis for the cross-attention and ControlNet feature-map outputs
produced by the selected inference-output folder.

The script does not re-run diffusion inference. It recursively scans only the
folder selected in INPUT_ROOT, deduplicates metadata, reads saved .npy maps,
runs/reuses Mask2Former segmentation for generated street-view images, and
produces:

1. cross_attention_map_metrics.csv
   - normalized spatial entropy
   - top-k attention mass
   - centroid and spatial spread
   - road/building/background attention mass and enrichment

2. cross_attention_level_change.csv
   - low-vs-medium, medium-vs-high, and low-vs-high JSD/cosine/Pearson
   - calculated for target phrases and cue aggregates

3. cross_attention_target_cue_alignment.csv
   - target-phrase vs cue-aggregate cosine/Pearson/JSD for full models

4. control_map_metrics.csv
   - feature-map region response and foreground enrichment
   - existing feature norms and delta metrics are retained from metadata

5. control_level_stability.csv
   - pairwise low/medium/high spatial similarity for matched blocks/timesteps

6. mechanism_summary_main_timestep.csv
   - descriptive mean/std summaries at MAIN_TIMESTEP_FRACTION

7. qualitative_panels/
   - target cross-attention and middle-delta comparison panels at the main
     timestep, arranged by experiment and level for each image/dimension.

8. mask2former_semantic_analysis/
   - generated-image street-element segmentation cache and overlays
   - cross-attention and ControlNet response by semantic street element
   - experiment/checkpoint summary CSVs and polar response figures.

The code supports both directory structures:
  .../sweep/<experiment>/<checkpoint>/...
  .../sweep/<experiment>/<checkpoint>/cross_attention/...
  .../sweep/<experiment>/<checkpoint>/control_analysis/...

No argparse is used. Edit only the USER SETTINGS section.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# 0. USER SETTINGS
# ============================================================

# Analyze only this folder in one run. The folder may be the sweep folder
# itself, for example:
#   .../inference_outputs_multidim_ablation/sweep
# or:
#   .../inference_outputs_untrained_baseline/sweep
INPUT_ROOT = Path(
    "/home/yuebing/GenAI_density_urbanization/urban_data/"
    "inference_outputs_multidim_ablation/sweep"
)

# Usually leave this as None. When None, an input ending in /sweep uses the
# parent directory name as RUN_NAME; otherwise INPUT_ROOT.name is used.
RUN_NAME_OVERRIDE: Optional[str] = None

# Each input obtains an independent output subfolder under this directory.
OUTPUT_BASE_DIR = Path(
    "/home/yuebing/GenAI_density_urbanization/urban_data/"
    "mechanism_analysis_by_input"
)

RUN_NAME = (
    RUN_NAME_OVERRIDE
    if RUN_NAME_OVERRIDE
    else (
        INPUT_ROOT.parent.name
        if INPUT_ROOT.name.lower() == "sweep"
        else INPUT_ROOT.name
    )
)
OUTPUT_DIR = OUTPUT_BASE_DIR / RUN_NAME

# Main paper analysis uses the middle denoising stage. All stages are still
# retained in detailed CSV outputs.
MAIN_TIMESTEP_FRACTION = 0.60
TIMESTEP_TOLERANCE = 0.06

LEVEL_ORDER = ["low", "medium", "high"]
EXPERIMENT_ORDER = ["baseline", "view_only", "view_level_only", "full"]

# Cross-attention settings.
# Keep token rows as a fallback for older outputs whose complete target phrase
# was not captured. Phrase-average rows are always preferred when available.
ATTENTION_UNITS = ["phrase_average", "token"]
ATTENTION_LAYER_GROUPS: Optional[List[str]] = None  # None = keep all groups
TOP_MASS_FRACTIONS = [0.01, 0.05, 0.10]

# Performance settings. Saved maps are usually 512x512 even though the original
# attention resolution is much smaller. Downsampling before metric computation
# greatly reduces memory use without changing the qualitative spatial pattern.
ANALYSIS_MAP_SIZE = 128

# Analyze complete target/cue phrase maps. Keep only the target-dimension token
# as a fallback for older outputs that did not save a target phrase average.
ATTENTION_PHRASES_ONLY = True

# Show progress bars for potentially large metadata tables.
SHOW_PROGRESS = True

# Optional semantic masks, useful for beautiful cues such as vegetation and
# architecture. Leave empty to use only road/building/background masks derived
# from the saved view_hint. Supported binary-mask layouts include:
#   root/<image_id>/<class>.png
#   root/<image_id>_<class>.png
#   root/<class>/<image_id>.png
SEMANTIC_MASK_ROOTS: Dict[str, Path] = {
    # "single": Path("/path/to/single/semantic_masks"),
    # "multi": Path("/path/to/multi/semantic_masks"),
}
SEMANTIC_CLASSES = ["vegetation", "building", "road", "sky"]

# Control analysis settings.
# Detailed metrics retain both residuals and middle deltas. Main panels use the
# middle delta because it is directly comparable across experiments.
CONTROL_ANALYSIS_TYPES = ["control_residual", "control_delta_middle"]
PANEL_CONTROL_ANALYSIS_TYPE = "control_delta_middle"

# Qualitative panels.
SAVE_QUALITATIVE_PANELS = True
PANEL_MAX_IMAGES_PER_RUN: Optional[int] = None  # None = all available images
PANEL_FIG_DPI = 180

# Missing/corrupt maps are reported and skipped rather than terminating the run.
STRICT_FILE_CHECK = False

# ------------------------------------------------------------
# Mask2Former street-element response analysis
# ------------------------------------------------------------
ENABLE_MASK2FORMER_ANALYSIS = True

# A street-scene semantic-segmentation checkpoint. The first run downloads the
# model through Hugging Face unless it is already available in the local cache.
MASK2FORMER_MODEL_NAME = (
    "facebook/mask2former-swin-large-mapillary-vistas-semantic"
)

# None = automatically use CUDA when available; otherwise set "cuda" or "cpu".
MASK2FORMER_DEVICE: Optional[str] = None

# metadata_sweep.csv produced by the current inference scripts stores generated
# images in generated_path. Additional names are retained for compatibility.
GENERATED_IMAGE_PATH_COLUMNS = [
    "generated_path",
    "generated_image_path",
    "output_image_path",
    "output_path",
    "result_image_path",
    "sample_path",
    "saved_image_path",
]

# Analyze semantic response at the same middle denoising stage used by the main
# mechanism summary. Set False to retain all saved denoising stages.
MASK2FORMER_MAIN_TIMESTEP_ONLY = True

# None = retain all selected attention layer groups and average them at the
# sample-summary stage. Target phrase maps are preferred, with the target token
# used as a fallback for older/raw-prompt outputs.
MASK2FORMER_ATTENTION_LAYER_GROUPS: Optional[List[str]] = None

# Analyze the directly comparable middle-delta ControlNet response. Change or
# extend this list when residual maps should also be summarized.
MASK2FORMER_CONTROL_ANALYSIS_TYPES = ["control_delta_middle"]
MASK2FORMER_CONTROL_BLOCK_NAMES: Optional[List[str]] = None

# Cache segmentation maps so rerunning the script does not repeat Mask2Former
# inference for images that have already been segmented.
MASK2FORMER_SAVE_SEGMENTATION_OVERLAYS = True
MASK2FORMER_OVERLAY_ALPHA = 0.55

# Small/rare semantic regions remain in CSV files but can be omitted from the
# checkpoint-level polar figure.
MASK2FORMER_MIN_PIXEL_SHARE_FOR_CHART = 0.005
MASK2FORMER_MIN_SAMPLES_FOR_CHART = 3
# The radial axis is chosen dynamically from the observed values and then
# rounded up to the nearest 0.5. This setting acts as a soft upper cap.
MASK2FORMER_RADIAL_MAX_ENRICHMENT = 3.0
MASK2FORMER_FIG_DPI = 220

# Separate semantic radial figures are generated for cross-attention and
# ControlNet analysis. Within each figure, low/medium/high levels are shown
# as three orange bars per semantic category.
# Default level palette, used by the dimension-aggregated ControlNet chart.
MASK2FORMER_LEVEL_COLORS: Dict[str, str] = {
    "low": "#FDC7A7",
    "medium": "#F9A879",
    "high": "#F5751C",
}

# Cross-attention charts are generated separately for each target dimension.
# Each dimension therefore uses its own low/medium/high sequential palette.
MASK2FORMER_DIMENSION_LEVEL_COLORS: Dict[str, Dict[str, str]] = {
    "beautiful": {
        "low": "#FDD0A2",
        "medium": "#FD8D3C",
        "high": "#D94801",
    },
    "safety": {
        "low": "#C7E9C0",
        "medium": "#74C476",
        "high": "#238B45",
    },
    "lively": {
        "low": "#FEC7D7",
        "medium": "#F768A1",
        "high": "#AE017E",
    },
    "depressing": {
        "low": "#C6DBEF",
        "medium": "#6BAED6",
        "high": "#2171B5",
    },
    "boring": {
        "low": "#D9D9D9",
        "medium": "#969696",
        "high": "#525252",
    },
    "wealthy": {
        "low": "#FFF7BC",
        "medium": "#FEC44F",
        "high": "#D95F0E",
    },
}

# Exactly three times the original font sizes: 28/20/20/18 -> 84/60/60/54.
MASK2FORMER_RADIAL_FONT_SCALE = 3.0
MASK2FORMER_RADIAL_TITLE_FONTSIZE = int(28 * MASK2FORMER_RADIAL_FONT_SCALE)
MASK2FORMER_RADIAL_LABEL_FONTSIZE = int(20 * MASK2FORMER_RADIAL_FONT_SCALE)
MASK2FORMER_RADIAL_TICK_FONTSIZE = int(20 * MASK2FORMER_RADIAL_FONT_SCALE)
MASK2FORMER_RADIAL_LEGEND_FONTSIZE = int(18 * MASK2FORMER_RADIAL_FONT_SCALE)

# A larger canvas prevents the 3x text from overlapping excessively.
MASK2FORMER_RADIAL_FIGSIZE = (24, 24)

# Keep text as editable text in SVG instead of converting glyphs to paths.
plt.rcParams["svg.fonttype"] = "none"

# Ordered semantic-group rules. The first matching group wins.
MASK2FORMER_STREET_ELEMENT_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("sky", ("sky",)),
    (
        "road",
        (
            "road",
            "lane marking",
            "crosswalk",
            "pothole",
            "rail track",
        ),
    ),
    (
        "sidewalk",
        (
            "sidewalk",
            "curb",
            "pedestrian area",
        ),
    ),
    (
        "building",
        (
            "building",
            "house",
            "garage",
            "bridge",
            "tunnel",
            "wall",
        ),
    ),
    (
        "vegetation",
        (
            "vegetation",
            "tree",
            "grass",
            "plant",
            "flower",
            "terrain",
        ),
    ),
    ("person", ("person", "rider")),
    (
        "vehicle",
        (
            "car",
            "truck",
            "bus",
            "vehicle",
            "motorcycle",
            "bicycle",
            "caravan",
            "trailer",
            "boat",
            "train",
        ),
    ),
    (
        "traffic_control",
        (
            "traffic sign",
            "traffic light",
            "stop sign",
            "information sign",
        ),
    ),
    (
        "street_furniture",
        (
            "street light",
            "pole",
            "bench",
            "fire hydrant",
            "mailbox",
            "trash can",
            "phone booth",
            "banner",
            "billboard",
        ),
    ),
    (
        "barrier_fence",
        (
            "fence",
            "guard rail",
            "barrier",
            "gate",
        ),
    ),
    ("water", ("water", "river", "sea", "lake")),
]

MASK2FORMER_SEMANTIC_GROUP_ORDER = [
    "sky",
    "building",
    "vegetation",
    "road",
    "sidewalk",
    "traffic_control",
    "street_furniture",
    "barrier_fence",
    "vehicle",
    "person",
    "water",
    "other",
]

EPS = 1e-12


# ============================================================
# 1. Generic helpers
# ============================================================


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_level(value) -> str:
    s = str(value).strip().lower().replace("_", " ").replace("-", " ")
    mapping = {
        "0": "low",
        "1": "low",
        "2": "medium",
        "3": "high",
        "very low": "very low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "very high": "very high",
    }
    return mapping.get(s, s)


def safe_float(value, default: float = np.nan) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def nearest_main_timestep_mask(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return (vals - MAIN_TIMESTEP_FRACTION).abs() <= TIMESTEP_TOLERANCE


def existing_path(value, metadata_path: Optional[Path] = None) -> Optional[Path]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None

    p = Path(text)
    if p.exists():
        return p

    # Fallbacks for moved output directories.
    candidates: List[Path] = []
    if metadata_path is not None:
        candidates.extend(
            [
                metadata_path.parent / p.name,
                metadata_path.parent / text,
            ]
        )
        # Search a limited local subtree by basename.
        try:
            candidates.extend(list(metadata_path.parent.rglob(p.name))[:5])
        except Exception:
            pass

    for cand in candidates:
        if cand.exists():
            return cand

    if STRICT_FILE_CHECK:
        raise FileNotFoundError(f"Map file not found: {text}")
    return None


def read_map(value, metadata_path: Optional[Path] = None) -> Optional[np.ndarray]:
    p = existing_path(value, metadata_path)
    if p is None:
        return None
    try:
        # mmap avoids eagerly copying the full saved array. Most inference
        # outputs store 512x512 float maps; metrics are computed on a compact
        # analysis grid to keep memory bounded.
        arr = np.load(p, mmap_mode="r")
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            warnings.warn(f"Skip non-2D map: {p}, shape={arr.shape}")
            return None

        target = int(ANALYSIS_MAP_SIZE)
        if target > 0 and arr.shape != (target, target):
            arr = cv2.resize(
                np.asarray(arr, dtype=np.float32),
                (target, target),
                interpolation=cv2.INTER_AREA,
            )
        else:
            arr = np.asarray(arr, dtype=np.float32)

        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.ascontiguousarray(arr, dtype=np.float32)
    except Exception as exc:
        warnings.warn(f"Cannot read map {p}: {exc}")
        if STRICT_FILE_CHECK:
            raise
        return None


def resize_map(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr.astype(np.float32, copy=False)
    return cv2.resize(
        arr.astype(np.float32),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )


def nonnegative(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    amin = float(np.min(arr))
    if amin < 0:
        arr = arr - amin
    return np.maximum(arr, 0.0)


def probability_map(arr: np.ndarray) -> np.ndarray:
    x = nonnegative(arr)
    total = float(x.sum())
    if total <= EPS:
        return np.full(x.shape, 1.0 / x.size, dtype=np.float64)
    return x / total


def minmax_map(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin + EPS:
        return np.zeros(x.shape, dtype=np.float64)
    return (x - xmin) / (xmax - xmin)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    b = resize_map(b, a.shape)
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > EPS else np.nan


def pearson_similarity(a: np.ndarray, b: np.ndarray) -> float:
    b = resize_map(b, a.shape)
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    if np.std(x) <= EPS or np.std(y) <= EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    b = resize_map(b, a.shape)
    p = probability_map(a).ravel()
    q = probability_map(b).ravel()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + EPS) / (m + EPS)))
    kl_qm = np.sum(q * np.log((q + EPS) / (m + EPS)))
    return float(0.5 * (kl_pm + kl_qm) / math.log(2.0))


def normalized_entropy(arr: np.ndarray) -> float:
    p = probability_map(arr).ravel()
    n = p.size
    if n <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p + EPS)) / math.log(n))


def top_mass(arr: np.ndarray, fraction: float) -> float:
    p = probability_map(arr).ravel()
    k = max(1, int(math.ceil(float(fraction) * p.size)))
    return float(np.partition(p, -k)[-k:].sum())


def spatial_moments(arr: np.ndarray) -> Dict[str, float]:
    p = probability_map(arr)
    h, w = p.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x01 = xx / max(w - 1, 1)
    y01 = yy / max(h - 1, 1)
    cx = float(np.sum(p * x01))
    cy = float(np.sum(p * y01))
    spread = float(
        math.sqrt(np.sum(p * ((x01 - cx) ** 2 + (y01 - cy) ** 2)))
    )
    return {
        "centroid_x": cx,
        "centroid_y": cy,
        "spatial_spread": spread,
    }


# ============================================================
# 2. Metadata discovery
# ============================================================


@dataclass
class RunTables:
    run_name: str
    root: Path
    attention: pd.DataFrame
    control: pd.DataFrame
    sweep: pd.DataFrame


def read_metadata_files(root: Path, filename: str, run_name: str) -> pd.DataFrame:
    files = sorted(root.rglob(filename))
    frames: List[pd.DataFrame] = []

    for file in files:
        try:
            df = pd.read_csv(file)
        except Exception as exc:
            warnings.warn(f"Cannot read {file}: {exc}")
            continue
        if df.empty:
            continue
        df = df.copy()
        df["run_name"] = run_name
        df["metadata_source"] = str(file)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


def deduplicate_attention(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = [
        c
        for c in [
            "run_name",
            "experiment",
            "ckpt_tag",
            "image_id",
            "target_dim",
            "target_level",
            "ddim_step_index",
            "original_timestep_value",
            "attention_layer_group",
            "attention_unit",
            "token_index",
            "matched_keyword",
            "attention_npy_path",
        ]
        if c in df.columns
    ]
    return df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)


def deduplicate_control(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = [
        c
        for c in [
            "run_name",
            "experiment",
            "ckpt_tag",
            "image_id",
            "target_dim",
            "target_level",
            "analysis_type",
            "block_index",
            "block_name",
            "ddim_step_index",
            "original_timestep_value",
            "control_map_npy_path",
        ]
        if c in df.columns
    ]
    return df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)


def load_run(run_name: str, root: Path) -> Optional[RunTables]:
    if not root.exists():
        warnings.warn(f"Skip missing inference root: {root}")
        return None

    attention = deduplicate_attention(
        read_metadata_files(root, "cross_attention_metadata.csv", run_name)
    )
    control = deduplicate_control(
        read_metadata_files(root, "control_analysis_metadata.csv", run_name)
    )
    sweep = read_metadata_files(root, "metadata_sweep.csv", run_name)

    print(
        f"[LOAD] {run_name}: attention={len(attention)}, "
        f"control={len(control)}, sweep={len(sweep)}"
    )
    return RunTables(run_name, root, attention, control, sweep)


# ============================================================
# 3. Canonical view-hint masks
# ============================================================


def build_canonical_view_hint_lookup(tables: Sequence[RunTables]) -> Dict[Tuple[str, str], Path]:
    lookup: Dict[Tuple[str, str], Path] = {}

    for table in tables:
        candidates = []
        for df in [table.sweep, table.attention, table.control]:
            if df.empty or "view_hint_path" not in df.columns or "image_id" not in df.columns:
                continue
            candidates.append(df[["image_id", "view_hint_path", "metadata_source"]].copy())

        if not candidates:
            continue

        merged = pd.concat(candidates, ignore_index=True).drop_duplicates()
        for _, row in merged.iterrows():
            image_id = str(row.get("image_id", ""))
            meta_path = Path(str(row.get("metadata_source", "")))
            p = existing_path(row.get("view_hint_path", ""), meta_path)
            if image_id and p is not None:
                lookup.setdefault((table.run_name, image_id), p)

    return lookup


def find_semantic_mask(run_name: str, image_id: str, class_name: str) -> Optional[Path]:
    root = SEMANTIC_MASK_ROOTS.get(run_name)
    if root is None or not Path(root).exists():
        return None
    root = Path(root)
    candidates = [
        root / image_id / f"{class_name}.png",
        root / f"{image_id}_{class_name}.png",
        root / class_name / f"{image_id}.png",
        root / image_id / f"{class_name}.jpg",
        root / f"{image_id}_{class_name}.jpg",
        root / class_name / f"{image_id}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# The same four view hints are reused by thousands of attention/control rows.
# Cache masks by file and analysis shape so each image is decoded only once.
_VIEW_MASK_CACHE: Dict[Tuple[str, int, int], Dict[str, np.ndarray]] = {}
_SEMANTIC_MASK_CACHE: Dict[Tuple[str, str, int, int], Dict[str, np.ndarray]] = {}


def read_optional_semantic_masks(
    run_name: str,
    image_id: str,
    shape: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    cache_key = (str(run_name), str(image_id), int(shape[0]), int(shape[1]))
    cached = _SEMANTIC_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    masks: Dict[str, np.ndarray] = {}
    for class_name in SEMANTIC_CLASSES:
        p = find_semantic_mask(run_name, image_id, class_name)
        if p is None:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        masks[f"semantic_{class_name}"] = img > 0

    _SEMANTIC_MASK_CACHE[cache_key] = masks
    return masks


def read_view_hint_masks(path: Path, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    cache_key = (str(Path(path).resolve()), int(shape[0]), int(shape[1]))
    cached = _VIEW_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        z = np.zeros(shape, dtype=bool)
        masks = {"road": z, "building": z, "background": ~z, "foreground": z}
        _VIEW_MASK_CACHE[cache_key] = masks
        return masks

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    arr = img.astype(np.float32, copy=False)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Faster and allocation-light equivalent of min(255-arr)>10.
    nonwhite = (r < 245.0) | (g < 245.0) | (b < 245.0)
    road = nonwhite & (r > b + 25.0) & (g > b + 5.0)
    building = nonwhite & (b > r + 20.0) & (b > g + 10.0)
    foreground = road | building
    background = ~foreground

    masks = {
        "road": road,
        "building": building,
        "foreground": foreground,
        "background": background,
    }
    _VIEW_MASK_CACHE[cache_key] = masks
    return masks


def region_metrics(arr: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    p = probability_map(arr)
    mm = minmax_map(arr)
    whole_mean = float(mm.mean()) + EPS

    out: Dict[str, float] = {}
    for name, mask in masks.items():
        if mask.shape != arr.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (arr.shape[1], arr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if not mask.any():
            out[f"{name}_mass"] = np.nan
            out[f"{name}_mean"] = np.nan
            out[f"{name}_enrichment"] = np.nan
            continue

        out[f"{name}_mass"] = float(p[mask].sum())
        mean_val = float(mm[mask].mean())
        out[f"{name}_mean"] = mean_val
        out[f"{name}_enrichment"] = mean_val / whole_mean

    fg_mean = out.get("foreground_mean", np.nan)
    bg_mean = out.get("background_mean", np.nan)
    out["foreground_background_ratio"] = (
        float(fg_mean / (bg_mean + EPS))
        if np.isfinite(fg_mean) and np.isfinite(bg_mean)
        else np.nan
    )
    return out


# ============================================================
# 4. Cross-attention analysis
# ============================================================


def attention_role(matched_keyword: str) -> str:
    key = str(matched_keyword)
    if key.startswith("target_score_phrase:"):
        return "target_phrase"
    if key.startswith("target_level_phrase:"):
        return "target_phrase"
    if key.startswith("cue_phrase:"):
        return "cue_phrase"
    return "keyword_or_token"


def prepare_attention_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    if "attention_unit" in out.columns:
        out = out[out["attention_unit"].isin(ATTENTION_UNITS)].copy()
    if ATTENTION_LAYER_GROUPS is not None and "attention_layer_group" in out.columns:
        out = out[out["attention_layer_group"].isin(ATTENTION_LAYER_GROUPS)].copy()

    if "target_level" not in out.columns:
        out["target_level"] = ""
    if "matched_keyword" not in out.columns:
        out["matched_keyword"] = ""
    if "target_dim" not in out.columns:
        out["target_dim"] = ""

    out["target_level"] = out["target_level"].map(normalize_level)
    out["attention_role"] = out["matched_keyword"].map(attention_role)

    if ATTENTION_PHRASES_ONLY and "attention_unit" in out.columns:
        unit = out["attention_unit"].astype(str)
        matched = out["matched_keyword"].astype(str).str.lower().str.strip()
        target = out["target_dim"].astype(str).str.lower().str.strip()
        keep = (unit == "phrase_average") | ((unit == "token") & (matched == target))
        out = out[keep].copy()

    return out.reset_index(drop=True)


def compute_attention_map_metrics(
    df: pd.DataFrame,
    view_lookup: Dict[Tuple[str, str], Path],
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    rows: List[Dict] = []
    map_cache: Dict[int, np.ndarray] = {}

    iterator = df.iterrows()
    if SHOW_PROGRESS:
        iterator = tqdm(iterator, total=len(df), desc="attention metrics")

    for idx, row in iterator:
        meta_path = Path(str(row.get("metadata_source", "")))
        arr = read_map(row.get("attention_npy_path", ""), meta_path)
        if arr is None:
            continue
        map_cache[idx] = arr

        record = row.to_dict()
        record.update(
            {
                "normalized_entropy": normalized_entropy(arr),
                **spatial_moments(arr),
                "map_min": float(np.min(arr)),
                "map_max": float(np.max(arr)),
                "map_mean": float(np.mean(arr)),
                "map_std": float(np.std(arr)),
            }
        )
        for frac in TOP_MASS_FRACTIONS:
            record[f"top_{int(frac * 100):02d}pct_mass"] = top_mass(arr, frac)

        run_name = str(row.get("run_name", ""))
        image_id = str(row.get("image_id", ""))
        key = (run_name, image_id)
        view_path = view_lookup.get(key)
        masks: Dict[str, np.ndarray] = {}
        if view_path is not None:
            masks.update(read_view_hint_masks(view_path, arr.shape))
            record["canonical_view_hint_path"] = str(view_path)
        else:
            record["canonical_view_hint_path"] = ""
        masks.update(read_optional_semantic_masks(run_name, image_id, arr.shape))
        if masks:
            record.update(region_metrics(arr, masks))

        record["map_row_id"] = idx
        rows.append(record)

    return pd.DataFrame(rows), map_cache


def aggregate_cue_maps(
    df: pd.DataFrame,
    map_cache: Dict[int, np.ndarray],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    if df.empty:
        return pd.DataFrame(), {}

    cue_df = df[df["attention_role"] == "cue_phrase"].copy()
    if cue_df.empty:
        return pd.DataFrame(), {}

    group_cols = [
        c
        for c in [
            "run_name",
            "experiment",
            "ckpt_tag",
            "image_id",
            "target_dim",
            "target_level",
            "ddim_step_index",
            "original_timestep_value",
            "timestep_fraction",
            "attention_layer_group",
        ]
        if c in cue_df.columns
    ]

    rows: List[Dict] = []
    agg_maps: Dict[str, np.ndarray] = {}

    for group_key, group in cue_df.groupby(group_cols, dropna=False):
        maps = [map_cache[i] for i in group.index if i in map_cache]
        if not maps:
            continue
        shape = maps[0].shape
        aligned = [probability_map(resize_map(x, shape)) for x in maps]
        agg = np.mean(np.stack(aligned, axis=0), axis=0).astype(np.float32)

        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        record = dict(zip(group_cols, key_values))
        agg_id = "|".join(str(record.get(c, "")) for c in group_cols)
        agg_maps[agg_id] = agg
        record.update(
            {
                "aggregate_map_id": agg_id,
                "attention_role": "cue_aggregate",
                "n_cue_phrases": int(len(maps)),
                "normalized_entropy": normalized_entropy(agg),
                **spatial_moments(agg),
            }
        )
        rows.append(record)

    return pd.DataFrame(rows), agg_maps


def build_target_map_lookup(
    df: pd.DataFrame,
    map_cache: Dict[int, np.ndarray],
) -> Dict[Tuple, np.ndarray]:
    """Build one target map per matched sample/step/layer.

    Complete phrase-average maps are preferred. For older inference outputs in
    which the raw-score phrase matcher was not aligned with the prompt, the
    function falls back to token maps matched to the target dimension name.
    """
    lookup: Dict[Tuple, np.ndarray] = {}
    key_cols = [
        "run_name",
        "experiment",
        "ckpt_tag",
        "image_id",
        "target_dim",
        "target_level",
        "ddim_step_index",
        "original_timestep_value",
        "attention_layer_group",
    ]

    for key, group in df.groupby(key_cols, dropna=False):
        phrase_group = group[group["attention_role"] == "target_phrase"]
        chosen = phrase_group

        if chosen.empty:
            target_dim = str(group.iloc[0].get("target_dim", "")).lower()
            matched = group.get("matched_keyword", pd.Series(index=group.index, dtype=str)).astype(str).str.lower()
            token_unit = group.get("attention_unit", pd.Series(index=group.index, dtype=str)).astype(str)
            chosen = group[(token_unit == "token") & (matched == target_dim)]

        maps = [map_cache[i] for i in chosen.index if i in map_cache]
        if not maps:
            continue
        shape = maps[0].shape
        aligned = [probability_map(resize_map(x, shape)) for x in maps]
        lookup[key] = np.mean(np.stack(aligned, axis=0), axis=0).astype(np.float32)

    return lookup


def compute_attention_level_change(
    df: pd.DataFrame,
    map_cache: Dict[int, np.ndarray],
    cue_agg_df: pd.DataFrame,
    cue_agg_maps: Dict[str, np.ndarray],
) -> pd.DataFrame:
    records: List[Dict] = []

    # Target-phrase maps.
    target_lookup = build_target_map_lookup(df, map_cache)
    base_cols = [
        "run_name",
        "experiment",
        "ckpt_tag",
        "image_id",
        "target_dim",
        "ddim_step_index",
        "original_timestep_value",
        "attention_layer_group",
    ]

    grouped: Dict[Tuple, Dict[str, np.ndarray]] = {}
    for key, arr in target_lookup.items():
        key_dict = dict(
            zip(
                [
                    "run_name",
                    "experiment",
                    "ckpt_tag",
                    "image_id",
                    "target_dim",
                    "target_level",
                    "ddim_step_index",
                    "original_timestep_value",
                    "attention_layer_group",
                ],
                key,
            )
        )
        base = tuple(key_dict[c] for c in base_cols)
        grouped.setdefault(base, {})[normalize_level(key_dict["target_level"])] = arr

    records.extend(
        pairwise_level_records(grouped, base_cols, "target_phrase")
    )

    # Cue aggregate maps.
    if not cue_agg_df.empty:
        cue_grouped: Dict[Tuple, Dict[str, np.ndarray]] = {}
        for _, row in cue_agg_df.iterrows():
            agg_id = str(row["aggregate_map_id"])
            arr = cue_agg_maps.get(agg_id)
            if arr is None:
                continue
            base = tuple(row.get(c, np.nan) for c in base_cols)
            cue_grouped.setdefault(base, {})[
                normalize_level(row.get("target_level", ""))
            ] = arr
        records.extend(
            pairwise_level_records(cue_grouped, base_cols, "cue_aggregate")
        )

    return pd.DataFrame(records)


def pairwise_level_records(
    grouped: Dict[Tuple, Dict[str, np.ndarray]],
    base_cols: Sequence[str],
    role: str,
) -> List[Dict]:
    pairs = [("low", "medium"), ("medium", "high"), ("low", "high")]
    records: List[Dict] = []

    for base, level_maps in grouped.items():
        base_record = dict(zip(base_cols, base))
        for level_a, level_b in pairs:
            a = level_maps.get(level_a)
            b = level_maps.get(level_b)
            if a is None or b is None:
                continue
            records.append(
                {
                    **base_record,
                    "attention_role": role,
                    "level_a": level_a,
                    "level_b": level_b,
                    "level_pair": f"{level_a}_vs_{level_b}",
                    "js_divergence": js_divergence(a, b),
                    "cosine_similarity": cosine_similarity(a, b),
                    "pearson_similarity": pearson_similarity(a, b),
                }
            )
    return records


def compute_target_cue_alignment(
    target_lookup: Dict[Tuple, np.ndarray],
    cue_agg_df: pd.DataFrame,
    cue_agg_maps: Dict[str, np.ndarray],
) -> pd.DataFrame:
    if cue_agg_df.empty:
        return pd.DataFrame()

    records: List[Dict] = []
    key_cols = [
        "run_name",
        "experiment",
        "ckpt_tag",
        "image_id",
        "target_dim",
        "target_level",
        "ddim_step_index",
        "original_timestep_value",
        "attention_layer_group",
    ]

    for _, row in cue_agg_df.iterrows():
        key = tuple(row.get(c, np.nan) for c in key_cols)
        target = target_lookup.get(key)
        cue = cue_agg_maps.get(str(row["aggregate_map_id"]))
        if target is None or cue is None:
            continue
        records.append(
            {
                **{c: row.get(c, np.nan) for c in key_cols},
                "target_cue_cosine": cosine_similarity(target, cue),
                "target_cue_pearson": pearson_similarity(target, cue),
                "target_cue_js_divergence": js_divergence(target, cue),
                "n_cue_phrases": row.get("n_cue_phrases", np.nan),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# 5. Control feature-map analysis
# ============================================================


def prepare_control_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "analysis_type" in out.columns:
        out = out[out["analysis_type"].isin(CONTROL_ANALYSIS_TYPES)].copy()
    if "target_level" not in out.columns:
        out["target_level"] = ""
    out["target_level"] = out["target_level"].map(normalize_level)
    return out.reset_index(drop=True)


def compute_control_map_metrics(
    df: pd.DataFrame,
    view_lookup: Dict[Tuple[str, str], Path],
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    rows: List[Dict] = []
    map_cache: Dict[int, np.ndarray] = {}

    iterator = df.iterrows()
    if SHOW_PROGRESS:
        iterator = tqdm(iterator, total=len(df), desc="control metrics")

    for idx, row in iterator:
        meta_path = Path(str(row.get("metadata_source", "")))
        arr = read_map(row.get("control_map_npy_path", ""), meta_path)
        if arr is None:
            continue
        map_cache[idx] = arr

        record = row.to_dict()
        record.update(
            {
                "map_normalized_entropy": normalized_entropy(arr),
                **spatial_moments(arr),
                "map_min": float(np.min(arr)),
                "map_max": float(np.max(arr)),
                "map_mean": float(np.mean(arr)),
                "map_std": float(np.std(arr)),
            }
        )

        run_name = str(row.get("run_name", ""))
        image_id = str(row.get("image_id", ""))
        key = (run_name, image_id)
        view_path = view_lookup.get(key)
        masks: Dict[str, np.ndarray] = {}
        if view_path is not None:
            masks.update(read_view_hint_masks(view_path, arr.shape))
            record["canonical_view_hint_path"] = str(view_path)
        else:
            record["canonical_view_hint_path"] = ""
        masks.update(read_optional_semantic_masks(run_name, image_id, arr.shape))
        if masks:
            record.update(region_metrics(arr, masks))

        record["map_row_id"] = idx
        rows.append(record)

    return pd.DataFrame(rows), map_cache


def compute_control_level_stability(
    df: pd.DataFrame,
    map_cache: Dict[int, np.ndarray],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = [
        c
        for c in [
            "run_name",
            "experiment",
            "ckpt_tag",
            "image_id",
            "target_dim",
            "analysis_type",
            "block_index",
            "block_name",
            "ddim_step_index",
            "original_timestep_value",
            "timestep_fraction",
        ]
        if c in df.columns
    ]

    pair_defs = [("low", "medium"), ("medium", "high"), ("low", "high")]
    records: List[Dict] = []

    for group_key, group in df.groupby(group_cols, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, key_values))
        level_maps: Dict[str, List[np.ndarray]] = {}

        for idx, row in group.iterrows():
            arr = map_cache.get(idx)
            if arr is None:
                continue
            level = normalize_level(row.get("target_level", ""))
            level_maps.setdefault(level, []).append(arr)

        level_mean: Dict[str, np.ndarray] = {}
        for level, maps in level_maps.items():
            shape = maps[0].shape
            aligned = [minmax_map(resize_map(x, shape)) for x in maps]
            level_mean[level] = np.mean(np.stack(aligned, axis=0), axis=0)

        pair_rows = []
        for level_a, level_b in pair_defs:
            a = level_mean.get(level_a)
            b = level_mean.get(level_b)
            if a is None or b is None:
                continue
            rec = {
                **base,
                "level_a": level_a,
                "level_b": level_b,
                "level_pair": f"{level_a}_vs_{level_b}",
                "cosine_similarity": cosine_similarity(a, b),
                "pearson_similarity": pearson_similarity(a, b),
                "js_divergence": js_divergence(a, b),
            }
            pair_rows.append(rec)
            records.append(rec)

        if pair_rows:
            records.append(
                {
                    **base,
                    "level_a": "all",
                    "level_b": "all",
                    "level_pair": "mean_all_pairs",
                    "cosine_similarity": float(
                        np.nanmean([r["cosine_similarity"] for r in pair_rows])
                    ),
                    "pearson_similarity": float(
                        np.nanmean([r["pearson_similarity"] for r in pair_rows])
                    ),
                    "js_divergence": float(
                        np.nanmean([r["js_divergence"] for r in pair_rows])
                    ),
                }
            )

    return pd.DataFrame(records)


# ============================================================
# 6. Summaries
# ============================================================


def mean_std_summary(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    existing_groups = [c for c in group_cols if c in df.columns]
    existing_values = [c for c in value_cols if c in df.columns]
    if not existing_values:
        return pd.DataFrame()

    grouped = df.groupby(existing_groups, dropna=False)[existing_values].agg(
        ["mean", "std", "count"]
    )
    grouped.columns = [
        f"{prefix}_{value}_{stat}" for value, stat in grouped.columns
    ]
    return grouped.reset_index()


def build_main_timestep_summary(
    attention_metrics: pd.DataFrame,
    attention_change: pd.DataFrame,
    target_cue_alignment: pd.DataFrame,
    control_metrics: pd.DataFrame,
    control_stability: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = ["run_name", "experiment", "target_dim"]
    pieces: List[pd.DataFrame] = []

    if not attention_metrics.empty and "timestep_fraction" in attention_metrics.columns:
        a = attention_metrics[nearest_main_timestep_mask(attention_metrics["timestep_fraction"])].copy()
        a = a[a["attention_role"].isin(["target_phrase", "cue_phrase"])]
        pieces.append(
            mean_std_summary(
                a,
                group_cols + ["attention_role"],
                [
                    "normalized_entropy",
                    "foreground_enrichment",
                    "road_enrichment",
                    "building_enrichment",
                ],
                "attn",
            )
        )

    if not attention_change.empty:
        ac = attention_change.copy()
        if "timestep_fraction" in ac.columns:
            ac = ac[nearest_main_timestep_mask(ac["timestep_fraction"])]
        ac = ac[ac["level_pair"] == "low_vs_high"]
        pieces.append(
            mean_std_summary(
                ac,
                group_cols + ["attention_role"],
                ["js_divergence", "cosine_similarity", "pearson_similarity"],
                "attn_low_high",
            )
        )

    if not target_cue_alignment.empty:
        tca = target_cue_alignment.copy()
        if "timestep_fraction" in tca.columns:
            tca = tca[nearest_main_timestep_mask(tca["timestep_fraction"])]
        pieces.append(
            mean_std_summary(
                tca,
                group_cols,
                [
                    "target_cue_cosine",
                    "target_cue_pearson",
                    "target_cue_js_divergence",
                ],
                "target_cue",
            )
        )

    if not control_metrics.empty and "timestep_fraction" in control_metrics.columns:
        cm = control_metrics[nearest_main_timestep_mask(control_metrics["timestep_fraction"])].copy()
        cm = cm[cm["analysis_type"] == PANEL_CONTROL_ANALYSIS_TYPE]
        pieces.append(
            mean_std_summary(
                cm,
                group_cols + ["analysis_type"],
                [
                    "foreground_enrichment",
                    "road_enrichment",
                    "building_enrichment",
                    "foreground_background_ratio",
                    "delta_cosine_distance",
                ],
                "control",
            )
        )

    if not control_stability.empty:
        cs = control_stability.copy()
        if "timestep_fraction" in cs.columns:
            cs = cs[nearest_main_timestep_mask(cs["timestep_fraction"])]
        cs = cs[
            (cs["analysis_type"] == PANEL_CONTROL_ANALYSIS_TYPE)
            & (cs["level_pair"] == "mean_all_pairs")
        ]
        pieces.append(
            mean_std_summary(
                cs,
                group_cols + ["analysis_type"],
                ["cosine_similarity", "pearson_similarity", "js_divergence"],
                "control_stability",
            )
        )

    valid = [x for x in pieces if not x.empty]
    if not valid:
        return pd.DataFrame()

    # Keep long-form summaries because attention_role/analysis_type differ.
    tagged = []
    for idx, piece in enumerate(valid):
        piece = piece.copy()
        piece["summary_block"] = idx
        tagged.append(piece)
    return pd.concat(tagged, ignore_index=True, sort=False)


# ============================================================
# 7. Qualitative panels
# ============================================================


def heatmap_rgb(arr: np.ndarray) -> np.ndarray:
    u8 = np.clip(minmax_map(arr) * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_panel(
    maps: Dict[Tuple[str, str], np.ndarray],
    title: str,
    out_path: Path,
) -> None:
    if not maps:
        return

    experiments = [x for x in EXPERIMENT_ORDER if any(k[0] == x for k in maps)]
    levels = [x for x in LEVEL_ORDER if any(k[1] == x for k in maps)]
    if not experiments or not levels:
        return

    fig, axes = plt.subplots(
        len(experiments),
        len(levels),
        figsize=(3.2 * len(levels), 2.8 * len(experiments)),
        squeeze=False,
    )

    for r, exp in enumerate(experiments):
        for c, level in enumerate(levels):
            ax = axes[r][c]
            arr = maps.get((exp, level))
            if arr is None:
                ax.axis("off")
                ax.set_title(f"{exp}\n{level}\nmissing")
                continue
            ax.imshow(heatmap_rgb(arr))
            ax.axis("off")
            ax.set_title(f"{exp}\n{level}")

    fig.suptitle(title)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=PANEL_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def build_attention_panels(
    attention_df: pd.DataFrame,
    map_cache: Dict[int, np.ndarray],
    out_dir: Path,
) -> None:
    if attention_df.empty:
        return

    df = attention_df[
        (attention_df["attention_role"] == "target_phrase")
        & nearest_main_timestep_mask(attention_df["timestep_fraction"])
    ].copy()
    if df.empty:
        return

    group_cols = ["run_name", "ckpt_tag", "image_id", "target_dim", "attention_layer_group"]
    count_per_run: Dict[str, int] = {}

    for group_key, group in df.groupby(group_cols, dropna=False):
        run_name, ckpt_tag, image_id, target_dim, layer_group = group_key
        if PANEL_MAX_IMAGES_PER_RUN is not None:
            count = count_per_run.get(str(run_name), 0)
            if count >= PANEL_MAX_IMAGES_PER_RUN:
                continue
            count_per_run[str(run_name)] = count + 1

        panel_maps: Dict[Tuple[str, str], np.ndarray] = {}
        for (exp, level), sub in group.groupby(["experiment", "target_level"], dropna=False):
            maps = [map_cache[i] for i in sub.index if i in map_cache]
            if not maps:
                continue
            shape = maps[0].shape
            panel_maps[(str(exp), normalize_level(level))] = np.mean(
                np.stack([probability_map(resize_map(x, shape)) for x in maps], axis=0),
                axis=0,
            )

        out_path = (
            out_dir
            / "cross_attention"
            / str(run_name)
            / str(ckpt_tag)
            / f"{image_id}_{target_dim}_{layer_group}_main_timestep.png"
        )
        save_panel(
            panel_maps,
            f"Target cross-attention | {run_name} | {image_id} | {target_dim} | {layer_group}",
            out_path,
        )


def build_control_panels(
    control_df: pd.DataFrame,
    map_cache: Dict[int, np.ndarray],
    out_dir: Path,
) -> None:
    if control_df.empty:
        return

    df = control_df[
        (control_df["analysis_type"] == PANEL_CONTROL_ANALYSIS_TYPE)
        & nearest_main_timestep_mask(control_df["timestep_fraction"])
    ].copy()
    if df.empty:
        return

    group_cols = ["run_name", "ckpt_tag", "image_id", "target_dim", "block_name"]
    count_per_run: Dict[str, int] = {}

    for group_key, group in df.groupby(group_cols, dropna=False):
        run_name, ckpt_tag, image_id, target_dim, block_name = group_key
        if PANEL_MAX_IMAGES_PER_RUN is not None:
            count = count_per_run.get(str(run_name), 0)
            if count >= PANEL_MAX_IMAGES_PER_RUN:
                continue
            count_per_run[str(run_name)] = count + 1

        panel_maps: Dict[Tuple[str, str], np.ndarray] = {}
        for (exp, level), sub in group.groupby(["experiment", "target_level"], dropna=False):
            maps = [map_cache[i] for i in sub.index if i in map_cache]
            if not maps:
                continue
            shape = maps[0].shape
            panel_maps[(str(exp), normalize_level(level))] = np.mean(
                np.stack([minmax_map(resize_map(x, shape)) for x in maps], axis=0),
                axis=0,
            )

        out_path = (
            out_dir
            / "control"
            / str(run_name)
            / str(ckpt_tag)
            / f"{image_id}_{target_dim}_{block_name}_main_timestep.png"
        )
        save_panel(
            panel_maps,
            f"Control middle delta | {run_name} | {image_id} | {target_dim}",
            out_path,
        )



# ============================================================
# 8. Mask2Former street-element response analysis
# ============================================================


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def sanitize_filename(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", clean_text(value))
    return text.strip("._") or "unknown"


def semantic_label_to_group(label: str) -> str:
    normalized = (
        str(label)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()

    for group_name, keywords in MASK2FORMER_STREET_ELEMENT_RULES:
        if any(keyword in normalized for keyword in keywords):
            return group_name
    return "other"


def infer_checkpoint_tag(row: Mapping[str, Any]) -> str:
    """Infer a stable checkpoint label from metadata fields or paths."""
    direct = clean_text(row.get("ckpt_tag", ""))
    if direct:
        return direct

    ckpt_path = clean_text(row.get("ckpt_path", ""))
    if ckpt_path:
        return Path(ckpt_path).stem

    metadata_source = clean_text(row.get("metadata_source", ""))
    experiment = clean_text(row.get("experiment", ""))
    if metadata_source:
        parts = list(Path(metadata_source).parts)
        if experiment and experiment in parts:
            pos = len(parts) - 1 - parts[::-1].index(experiment)
            if pos + 1 < len(parts):
                candidate = parts[pos + 1]
                if candidate not in {
                    "cross_attention",
                    "control_analysis",
                    "metadata_sweep.csv",
                    "cross_attention_metadata.csv",
                    "control_analysis_metadata.csv",
                }:
                    return candidate

        try:
            relative = Path(metadata_source).resolve().relative_to(
                INPUT_ROOT.resolve()
            )
            rel_parts = relative.parts
            if len(rel_parts) >= 2:
                return rel_parts[1]
        except Exception:
            pass

    return "unknown_ckpt"


def semantic_sample_key(row: Mapping[str, Any]) -> Tuple[str, ...]:
    return (
        clean_text(row.get("experiment", "")),
        infer_checkpoint_tag(row),
        clean_text(row.get("image_id", "")),
        clean_text(row.get("target_dim", "")),
        normalize_level(row.get("target_level", "")),
    )


def resolve_generated_path_from_row(row: Mapping[str, Any]) -> Optional[Path]:
    metadata_source_text = clean_text(row.get("metadata_source", ""))
    metadata_source = (
        Path(metadata_source_text) if metadata_source_text else None
    )

    for column in GENERATED_IMAGE_PATH_COLUMNS:
        if column not in row:
            continue
        path = existing_path(row.get(column, ""), metadata_source)
        if path is not None:
            return path
    return None


def build_generated_image_lookup(
    sweep_df: pd.DataFrame,
) -> Tuple[
    Dict[Tuple[str, ...], List[Path]],
    Dict[Tuple[str, ...], List[Path]],
    pd.DataFrame,
]:
    """Build exact checkpoint-aware and checkpoint-free generated-image lookups."""
    exact: Dict[Tuple[str, ...], Set[Path]] = defaultdict(set)
    fallback: Dict[Tuple[str, ...], Set[Path]] = defaultdict(set)
    rows: List[Dict[str, Any]] = []

    if sweep_df.empty:
        return {}, {}, pd.DataFrame()

    for _, row in sweep_df.iterrows():
        path = resolve_generated_path_from_row(row)
        if path is None:
            continue

        key = semantic_sample_key(row)
        fallback_key = (key[0], key[2], key[3], key[4])
        exact[key].add(path)
        fallback[fallback_key].add(path)

        rows.append(
            {
                "run_name": row.get("run_name", RUN_NAME),
                "experiment": key[0],
                "ckpt_tag": key[1],
                "image_id": key[2],
                "target_dim": key[3],
                "target_level": key[4],
                "generated_path": str(path),
                "metadata_source": row.get("metadata_source", ""),
            }
        )

    exact_sorted = {
        key: sorted(paths, key=lambda p: str(p))
        for key, paths in exact.items()
    }
    fallback_sorted = {
        key: sorted(paths, key=lambda p: str(p))
        for key, paths in fallback.items()
    }

    index_df = pd.DataFrame(rows).drop_duplicates()
    if not index_df.empty:
        candidate_counts = (
            index_df.groupby(
                [
                    "experiment",
                    "ckpt_tag",
                    "image_id",
                    "target_dim",
                    "target_level",
                ],
                dropna=False,
            )["generated_path"]
            .nunique()
            .rename("n_generated_candidates")
            .reset_index()
        )
        index_df = index_df.merge(
            candidate_counts,
            on=[
                "experiment",
                "ckpt_tag",
                "image_id",
                "target_dim",
                "target_level",
            ],
            how="left",
        )

    return exact_sorted, fallback_sorted, index_df


def resolve_generated_image_for_response(
    row: Mapping[str, Any],
    exact_lookup: Dict[Tuple[str, ...], List[Path]],
    fallback_lookup: Dict[Tuple[str, ...], List[Path]],
) -> Tuple[Optional[Path], int, str]:
    direct = resolve_generated_path_from_row(row)
    if direct is not None:
        return direct, 1, "direct_metadata_path"

    key = semantic_sample_key(row)
    exact_candidates = exact_lookup.get(key, [])
    if exact_candidates:
        return exact_candidates[0], len(exact_candidates), "exact_sample_key"

    fallback_key = (key[0], key[2], key[3], key[4])
    fallback_candidates = fallback_lookup.get(fallback_key, [])
    if len(fallback_candidates) == 1:
        return fallback_candidates[0], 1, "checkpoint_free_unique_key"
    if fallback_candidates:
        return (
            fallback_candidates[0],
            len(fallback_candidates),
            "checkpoint_free_ambiguous_first_sorted",
        )

    return None, 0, "not_found"


def mask2former_cache_key(image_path: Path) -> str:
    stat = image_path.stat()
    payload = (
        f"{image_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|"
        f"{MASK2FORMER_MODEL_NAME}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def deterministic_group_color(group_name: str) -> Tuple[int, int, int]:
    digest = hashlib.md5(group_name.encode("utf-8")).digest()
    return tuple(int(80 + value % 176) for value in digest[:3])


def save_mask2former_overlay(
    image_path: Path,
    grouped_map: np.ndarray,
    id_to_group: Dict[int, str],
    out_path: Path,
) -> None:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return

    if image_bgr.shape[:2] != grouped_map.shape:
        image_bgr = cv2.resize(
            image_bgr,
            (grouped_map.shape[1], grouped_map.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    group_rgb = np.zeros(
        (grouped_map.shape[0], grouped_map.shape[1], 3),
        dtype=np.uint8,
    )
    for group_id in np.unique(grouped_map):
        group_name = id_to_group.get(int(group_id), "other")
        group_rgb[grouped_map == group_id] = deterministic_group_color(
            group_name
        )

    group_bgr = cv2.cvtColor(group_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(
        image_bgr,
        1.0 - MASK2FORMER_OVERLAY_ALPHA,
        group_bgr,
        MASK2FORMER_OVERLAY_ALPHA,
        0.0,
    )
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), overlay)


def group_mask2former_segmentation(
    segmentation: np.ndarray,
    id2label: Dict[int, str],
) -> Tuple[np.ndarray, Dict[int, str], Dict[str, List[str]]]:
    group_names = list(MASK2FORMER_SEMANTIC_GROUP_ORDER)
    if "other" not in group_names:
        group_names.append("other")

    group_to_id = {name: idx for idx, name in enumerate(group_names)}
    id_to_group = {idx: name for name, idx in group_to_id.items()}
    grouped = np.full(
        segmentation.shape,
        group_to_id["other"],
        dtype=np.int16,
    )
    raw_labels_by_group: Dict[str, List[str]] = defaultdict(list)

    for raw_id in np.unique(segmentation):
        raw_label = id2label.get(int(raw_id), f"class_{int(raw_id)}")
        group_name = semantic_label_to_group(raw_label)
        grouped[segmentation == raw_id] = group_to_id.get(
            group_name,
            group_to_id["other"],
        )
        raw_labels_by_group[group_name].append(raw_label)

    raw_labels_by_group = {
        name: sorted(set(labels))
        for name, labels in raw_labels_by_group.items()
    }
    return grouped, id_to_group, raw_labels_by_group


def load_mask2former_runtime():
    try:
        import torch
        from PIL import Image
        from transformers import (
            AutoImageProcessor,
            Mask2FormerForUniversalSegmentation,
        )
    except ImportError as exc:
        raise ImportError(
            "Mask2Former analysis requires torch, Pillow and transformers. "
            "Install the missing package(s) in the active environment."
        ) from exc

    device = torch.device(
        MASK2FORMER_DEVICE
        if MASK2FORMER_DEVICE
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(
        f"[MASK2FORMER] loading {MASK2FORMER_MODEL_NAME} "
        f"on {device} ({dtype})"
    )
    processor = AutoImageProcessor.from_pretrained(
        MASK2FORMER_MODEL_NAME
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MASK2FORMER_MODEL_NAME,
        torch_dtype=dtype,
    )
    model = model.to(device).eval()

    id2label_raw = getattr(model.config, "id2label", {}) or {}
    id2label = {int(k): str(v) for k, v in id2label_raw.items()}
    return torch, Image, processor, model, device, id2label


def segment_generated_image(
    image_path: Path,
    runtime,
    cache_dir: Path,
    overlay_dir: Path,
    segmentation_memory_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    cache_id = mask2former_cache_key(image_path)
    if cache_id in segmentation_memory_cache:
        return segmentation_memory_cache[cache_id]

    torch, Image, processor, model, device, id2label = runtime
    raw_cache_path = cache_dir / f"{cache_id}_raw.npy"
    grouped_cache_path = cache_dir / f"{cache_id}_grouped.npy"
    meta_cache_path = cache_dir / f"{cache_id}.json"

    if raw_cache_path.exists() and grouped_cache_path.exists() and meta_cache_path.exists():
        segmentation = np.load(raw_cache_path, mmap_mode="r")
        grouped = np.load(grouped_cache_path, mmap_mode="r")
        meta = json.loads(meta_cache_path.read_text(encoding="utf-8"))
        id_to_group = {
            int(k): str(v) for k, v in meta["id_to_group"].items()
        }
        result = {
            "segmentation": np.asarray(segmentation),
            "grouped": np.asarray(grouped),
            "id_to_group": id_to_group,
            "raw_labels_by_group": meta.get("raw_labels_by_group", {}),
            "raw_cache_path": raw_cache_path,
            "grouped_cache_path": grouped_cache_path,
            "newly_inferred": False,
        }
        segmentation_memory_cache[cache_id] = result
        return result

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        inputs = processor(images=image, return_tensors="pt")
        prepared_inputs = {}
        for key, value in inputs.items():
            value = value.to(device)
            if torch.is_floating_point(value):
                value = value.to(dtype=next(model.parameters()).dtype)
            prepared_inputs[key] = value
        inputs = prepared_inputs

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        semantic = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(height, width)],
        )[0]
        segmentation = semantic.detach().cpu().numpy().astype(np.int16)

    grouped, id_to_group, raw_labels_by_group = (
        group_mask2former_segmentation(segmentation, id2label)
    )
    np.save(raw_cache_path, segmentation)
    np.save(grouped_cache_path, grouped)
    meta_cache_path.write_text(
        json.dumps(
            {
                "model_name": MASK2FORMER_MODEL_NAME,
                "image_path": str(image_path),
                "id_to_group": {
                    str(k): v for k, v in id_to_group.items()
                },
                "raw_labels_by_group": raw_labels_by_group,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if MASK2FORMER_SAVE_SEGMENTATION_OVERLAYS:
        save_mask2former_overlay(
            image_path,
            grouped,
            id_to_group,
            overlay_dir
            / f"{sanitize_filename(image_path.stem)}_{cache_id[:12]}.png",
        )

    result = {
        "segmentation": segmentation,
        "grouped": grouped,
        "id_to_group": id_to_group,
        "raw_labels_by_group": raw_labels_by_group,
        "raw_cache_path": raw_cache_path,
        "grouped_cache_path": grouped_cache_path,
        "newly_inferred": True,
    }
    segmentation_memory_cache[cache_id] = result
    return result


def response_probability_and_normalized_map(
    arr: np.ndarray,
    response_family: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(arr, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if response_family == "control_analysis":
        x = np.abs(x)
    else:
        xmin = float(np.min(x))
        if xmin < 0:
            x = x - xmin
        x = np.maximum(x, 0.0)

    total = float(x.sum())
    if total <= EPS:
        probability = np.full(x.shape, 1.0 / x.size, dtype=np.float64)
    else:
        probability = x / total

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin + EPS:
        normalized = np.zeros(x.shape, dtype=np.float64)
    else:
        normalized = (x - xmin) / (xmax - xmin)

    return probability, normalized


def semantic_region_response_metrics(
    response_map: np.ndarray,
    grouped_segmentation: np.ndarray,
    id_to_group: Dict[int, str],
    response_family: str,
) -> List[Dict[str, Any]]:
    if grouped_segmentation.shape != response_map.shape:
        grouped_segmentation = cv2.resize(
            grouped_segmentation.astype(np.int16),
            (response_map.shape[1], response_map.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    probability, normalized = response_probability_and_normalized_map(
        response_map,
        response_family,
    )
    total_pixels = int(grouped_segmentation.size)
    records: List[Dict[str, Any]] = []

    for group_id in np.unique(grouped_segmentation):
        mask = grouped_segmentation == group_id
        pixel_count = int(mask.sum())
        if pixel_count <= 0:
            continue

        pixel_share = pixel_count / total_pixels
        response_mass = float(probability[mask].sum())
        mean_response = float(normalized[mask].mean())
        enrichment = (
            float(response_mass / pixel_share)
            if pixel_share > EPS
            else np.nan
        )

        records.append(
            {
                "semantic_group": id_to_group.get(
                    int(group_id),
                    f"group_{int(group_id)}",
                ),
                "pixel_count": pixel_count,
                "pixel_share": pixel_share,
                "response_mass": response_mass,
                "response_minus_area_share": response_mass - pixel_share,
                "mean_normalized_response": mean_response,
                "response_enrichment": enrichment,
                "response_enrichment_log2": (
                    float(math.log2(enrichment + EPS))
                    if np.isfinite(enrichment)
                    else np.nan
                ),
            }
        )

    return records


def canonical_target_attention_entries(
    attention_df: pd.DataFrame,
    attention_maps: Dict[int, np.ndarray],
) -> List[Dict[str, Any]]:
    """Build one target-attention map per sample/timestep/layer group.

    Complete target-phrase maps are preferred. When they are unavailable, the
    target-dimension token is used, matching the fallback used by the original
    level-change analysis.
    """
    if attention_df.empty:
        return []

    df = attention_df.copy()
    if MASK2FORMER_MAIN_TIMESTEP_ONLY and "timestep_fraction" in df.columns:
        df = df[nearest_main_timestep_mask(df["timestep_fraction"])].copy()

    if (
        MASK2FORMER_ATTENTION_LAYER_GROUPS is not None
        and "attention_layer_group" in df.columns
    ):
        df = df[
            df["attention_layer_group"].isin(
                MASK2FORMER_ATTENTION_LAYER_GROUPS
            )
        ].copy()

    group_cols = [
        c
        for c in [
            "run_name",
            "experiment",
            "ckpt_tag",
            "image_id",
            "target_dim",
            "target_level",
            "ddim_step_index",
            "original_timestep_value",
            "timestep_fraction",
            "attention_layer_group",
        ]
        if c in df.columns
    ]

    entries: List[Dict[str, Any]] = []
    for _, group in df.groupby(group_cols, dropna=False):
        phrase_group = group[group["attention_role"] == "target_phrase"]
        chosen = phrase_group
        source = "target_phrase"

        if chosen.empty:
            target_dim = clean_text(group.iloc[0].get("target_dim", "")).lower()
            matched = group["matched_keyword"].astype(str).str.lower().str.strip()
            units = group["attention_unit"].astype(str)
            chosen = group[(units == "token") & (matched == target_dim)]
            source = "target_token_fallback"

        maps = [attention_maps[idx] for idx in chosen.index if idx in attention_maps]
        if not maps:
            continue

        shape = maps[0].shape
        aligned = [
            probability_map(resize_map(arr, shape))
            for arr in maps
        ]
        canonical_map = np.mean(
            np.stack(aligned, axis=0),
            axis=0,
        ).astype(np.float32)

        base = chosen.iloc[0].to_dict()
        base["response_map"] = canonical_map
        base["attention_map_source"] = source
        base["n_source_attention_maps"] = len(maps)
        base["ckpt_tag"] = infer_checkpoint_tag(base)
        entries.append(base)

    return entries


def control_response_entries(
    control_df: pd.DataFrame,
    control_maps: Dict[int, np.ndarray],
) -> List[Dict[str, Any]]:
    if control_df.empty:
        return []

    df = control_df.copy()
    if "analysis_type" in df.columns:
        df = df[
            df["analysis_type"].isin(
                MASK2FORMER_CONTROL_ANALYSIS_TYPES
            )
        ].copy()

    if MASK2FORMER_MAIN_TIMESTEP_ONLY and "timestep_fraction" in df.columns:
        df = df[nearest_main_timestep_mask(df["timestep_fraction"])].copy()

    if (
        MASK2FORMER_CONTROL_BLOCK_NAMES is not None
        and "block_name" in df.columns
    ):
        df = df[
            df["block_name"].isin(MASK2FORMER_CONTROL_BLOCK_NAMES)
        ].copy()

    entries: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        arr = control_maps.get(idx)
        if arr is None:
            continue
        base = row.to_dict()
        base["response_map"] = arr
        base["ckpt_tag"] = infer_checkpoint_tag(base)
        entries.append(base)
    return entries


def semantic_base_metadata(
    entry: Mapping[str, Any],
    response_family: str,
) -> Dict[str, Any]:
    keep = [
        "run_name",
        "experiment",
        "ckpt_tag",
        "image_id",
        "target_dim",
        "target_level",
        "ddim_step_index",
        "original_timestep_value",
        "timestep_fraction",
        "attention_layer_group",
        "attention_map_source",
        "n_source_attention_maps",
        "analysis_type",
        "block_index",
        "block_name",
        "metadata_source",
    ]
    record = {
        key: entry.get(key, np.nan)
        for key in keep
        if key in entry
    }
    record["response_family"] = response_family
    record["ckpt_tag"] = infer_checkpoint_tag(entry)
    record["target_level"] = normalize_level(
        entry.get("target_level", "")
    )
    return record


def build_semantic_sample_summary(
    detailed: pd.DataFrame,
) -> pd.DataFrame:
    if detailed.empty:
        return pd.DataFrame()

    group_cols = [
        c
        for c in [
            "run_name",
            "experiment",
            "ckpt_tag",
            "image_id",
            "target_dim",
            "target_level",
            "response_family",
            "semantic_group",
            "generated_path",
        ]
        if c in detailed.columns
    ]
    value_cols = [
        "pixel_share",
        "response_mass",
        "response_minus_area_share",
        "mean_normalized_response",
        "response_enrichment",
        "response_enrichment_log2",
    ]

    summary = (
        detailed.groupby(group_cols, dropna=False)[value_cols]
        .mean()
        .reset_index()
    )
    counts = (
        detailed.groupby(group_cols, dropna=False)
        .size()
        .rename("n_response_maps")
        .reset_index()
    )
    return summary.merge(counts, on=group_cols, how="left")


def build_semantic_checkpoint_summary(
    sample_summary: pd.DataFrame,
    extra_groups: Sequence[str] = (),
) -> pd.DataFrame:
    if sample_summary.empty:
        return pd.DataFrame()

    group_cols = [
        "run_name",
        "experiment",
        "ckpt_tag",
        *[c for c in extra_groups if c in sample_summary.columns],
        "response_family",
        "semantic_group",
    ]
    value_cols = [
        "pixel_share",
        "response_mass",
        "response_minus_area_share",
        "mean_normalized_response",
        "response_enrichment",
        "response_enrichment_log2",
    ]

    grouped = sample_summary.groupby(group_cols, dropna=False)[value_cols].agg(
        ["mean", "std", "median", "count"]
    )
    grouped.columns = [
        f"{metric}_{stat}" for metric, stat in grouped.columns
    ]
    out = grouped.reset_index()

    unique_stats = (
        sample_summary.groupby(group_cols, dropna=False)
        .agg(
            n_unique_images=("image_id", "nunique"),
            n_unique_target_dimensions=("target_dim", "nunique"),
            n_unique_levels=("target_level", "nunique"),
        )
        .reset_index()
    )
    return out.merge(unique_stats, on=group_cols, how="left")


def compute_dynamic_radial_outer_limit(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        observed_max = 1.0
    else:
        observed_max = max(1.0, float(np.max(finite)))
    outer = math.ceil((observed_max - 1e-12) * 2.0) / 2.0
    outer = max(1.0, outer)
    if np.isfinite(MASK2FORMER_RADIAL_MAX_ENRICHMENT):
        outer = min(outer, float(MASK2FORMER_RADIAL_MAX_ENRICHMENT))
    return outer


def response_family_display_name(response_family: str) -> str:
    mapping = {
        "cross_attention": "Cross-attention",
        "control_analysis": "Control analysis",
    }
    return mapping.get(str(response_family), str(response_family))


def semantic_polar_level_colors(target_dim: Optional[str]) -> Dict[str, str]:
    """Return the low/medium/high palette for one target dimension."""
    dim_key = clean_text(target_dim).lower()
    return MASK2FORMER_DIMENSION_LEVEL_COLORS.get(
        dim_key,
        MASK2FORMER_LEVEL_COLORS,
    )


def save_semantic_polar_chart(
    summary_df: pd.DataFrame,
    experiment: str,
    ckpt_tag: str,
    response_family: str,
    out_path: Path,
    target_dim: Optional[str] = None,
) -> None:
    sub = summary_df[
        (summary_df["experiment"].astype(str) == str(experiment))
        & (summary_df["ckpt_tag"].astype(str) == str(ckpt_tag))
        & (summary_df["response_family"].astype(str) == str(response_family))
    ].copy()
    if target_dim is not None and "target_dim" in sub.columns:
        sub = sub[sub["target_dim"].astype(str) == str(target_dim)].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(
        index="semantic_group",
        columns="target_level",
        values="response_enrichment_mean",
        aggfunc="first",
    )
    share = sub.groupby("semantic_group", dropna=False)["pixel_share_mean"].mean()
    count = sub.groupby("semantic_group", dropna=False)["response_enrichment_count"].max()

    groups = [
        group
        for group in MASK2FORMER_SEMANTIC_GROUP_ORDER
        if (
            group in pivot.index
            and float(share.get(group, 0.0)) >= MASK2FORMER_MIN_PIXEL_SHARE_FOR_CHART
            and float(count.get(group, 0.0)) >= MASK2FORMER_MIN_SAMPLES_FOR_CHART
        )
    ]
    if not groups:
        return

    levels = [level for level in LEVEL_ORDER if level in pivot.columns]
    if not levels:
        return

    values_by_level: Dict[str, np.ndarray] = {}
    all_values: List[float] = []
    for level in levels:
        values = np.array(
            [float(pivot.loc[group].get(level, np.nan)) for group in groups],
            dtype=float,
        )
        values_by_level[level] = values
        all_values.extend(values[np.isfinite(values)].tolist())
    all_values.append(1.0)

    outer_limit = compute_dynamic_radial_outer_limit(np.asarray(all_values, dtype=float))
    plotted_values_by_level = {
        level: np.clip(values, 0.0, outer_limit)
        for level, values in values_by_level.items()
    }

    n_groups = len(groups)
    theta = np.linspace(0.0, 2.0 * np.pi, n_groups, endpoint=False)
    category_width = 2.0 * np.pi / n_groups
    n_levels = len(levels)
    total_bar_span = category_width * 0.72
    bar_width = total_bar_span / max(n_levels, 1)
    offsets = (
        (np.arange(n_levels, dtype=float) - (n_levels - 1) / 2.0)
        * bar_width
    )

    level_colors = semantic_polar_level_colors(target_dim)

    fig = plt.figure(figsize=MASK2FORMER_RADIAL_FIGSIZE)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)

    for offset, level in zip(offsets, levels):
        ax.bar(
            theta + offset,
            np.nan_to_num(plotted_values_by_level[level], nan=0.0),
            width=bar_width * 0.92,
            alpha=0.90,
            color=level_colors.get(
                level,
                MASK2FORMER_LEVEL_COLORS.get(level, "#F9A879"),
            ),
            label=level.capitalize(),
        )

    reference_theta = np.linspace(0.0, 2.0 * np.pi, 360)
    ax.plot(
        reference_theta,
        np.ones_like(reference_theta),
        linestyle="--",
        linewidth=1.6 * MASK2FORMER_RADIAL_FONT_SCALE,
        color="#C98E5B",
        label="Area-proportional response (=1)",
    )

    ax.set_xticks(theta)
    ax.set_xticklabels(
        [group.replace("_", "\n") for group in groups],
        fontsize=MASK2FORMER_RADIAL_TICK_FONTSIZE,
    )
    ax.set_ylim(0.0, outer_limit)
    yticks = np.arange(0.5, outer_limit + 0.001, 0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [f"{tick:.1f}" for tick in yticks],
        fontsize=MASK2FORMER_RADIAL_TICK_FONTSIZE,
    )
    ax.set_ylabel(
        "Response enrichment",
        labelpad=36 * MASK2FORMER_RADIAL_FONT_SCALE,
        fontsize=MASK2FORMER_RADIAL_LABEL_FONTSIZE,
    )
    ax.grid(alpha=0.35)

    title_lines = [
        "Street-element response by Mask2Former semantic region",
        response_family_display_name(response_family),
    ]
    if target_dim is not None:
        title_lines.append(str(target_dim))
    fig.suptitle(
        "\n".join(title_lines),
        fontsize=MASK2FORMER_RADIAL_TITLE_FONTSIZE,
        y=0.98,
    )
    legend_title = (
        f"{str(target_dim).replace('_', ' ').title()} level"
        if target_dim is not None
        else None
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=max(2, len(levels) + 1),
        frameon=False,
        fontsize=MASK2FORMER_RADIAL_LEGEND_FONTSIZE,
        title=legend_title,
        title_fontsize=MASK2FORMER_RADIAL_LEGEND_FONTSIZE,
    )

    ensure_dir(out_path.parent)

    # Preserve the existing PNG output and automatically create a vector SVG.
    png_path = out_path.with_suffix(".png")
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(
        png_path,
        dpi=MASK2FORMER_FIG_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
    )
    plt.close(fig)


def run_mask2former_semantic_analysis(
    sweep_df: pd.DataFrame,
    attention_df: pd.DataFrame,
    attention_maps: Dict[int, np.ndarray],
    control_df: pd.DataFrame,
    control_maps: Dict[int, np.ndarray],
) -> Dict[str, pd.DataFrame]:
    semantic_root = ensure_dir(OUTPUT_DIR / "mask2former_semantic_analysis")
    cache_dir = ensure_dir(semantic_root / "segmentation_cache")
    overlay_dir = ensure_dir(semantic_root / "segmentation_overlays")
    by_ckpt_dir = ensure_dir(semantic_root / "by_experiment_ckpt")

    exact_lookup, fallback_lookup, path_index = (
        build_generated_image_lookup(sweep_df)
    )
    save_csv(
        path_index,
        semantic_root / "generated_image_path_index.csv",
    )
    if path_index.empty:
        raise RuntimeError(
            "Mask2Former analysis found no generated images. The current "
            "inference metadata normally uses the 'generated_path' column; "
            "check GENERATED_IMAGE_PATH_COLUMNS and metadata_sweep.csv."
        )

    attention_entries = canonical_target_attention_entries(
        attention_df,
        attention_maps,
    )
    control_entries = control_response_entries(
        control_df,
        control_maps,
    )
    print(
        f"[MASK2FORMER PREPARE] target attention maps={len(attention_entries)}, "
        f"control maps={len(control_entries)}"
    )

    runtime = load_mask2former_runtime()
    id2label = runtime[-1]
    (semantic_root / "mask2former_label_groups.json").write_text(
        json.dumps(
            {
                "model_name": MASK2FORMER_MODEL_NAME,
                "id2label": id2label,
                "mapped_groups": {
                    str(label_id): semantic_label_to_group(label)
                    for label_id, label in id2label.items()
                },
                "street_element_rules": [
                    [group, list(keywords)]
                    for group, keywords in MASK2FORMER_STREET_ELEMENT_RULES
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    detailed_records: List[Dict[str, Any]] = []
    missing_records: List[Dict[str, Any]] = []
    segmentation_index: Dict[str, Dict[str, Any]] = {}
    memory_cache: Dict[str, Dict[str, Any]] = {}

    tasks = [
        ("cross_attention", attention_entries),
        ("control_analysis", control_entries),
    ]
    total = sum(len(entries) for _, entries in tasks)
    progress = tqdm(
        total=total,
        desc="Mask2Former semantic response",
        disable=not SHOW_PROGRESS,
    )

    for response_family, entries in tasks:
        for entry in entries:
            progress.update(1)
            image_path, n_candidates, resolution_mode = (
                resolve_generated_image_for_response(
                    entry,
                    exact_lookup,
                    fallback_lookup,
                )
            )
            if image_path is None:
                missing_records.append(
                    {
                        **semantic_base_metadata(entry, response_family),
                        "reason": "generated_image_not_resolved",
                        "generated_path_resolution_mode": resolution_mode,
                    }
                )
                continue

            try:
                segmentation_result = segment_generated_image(
                    image_path,
                    runtime,
                    cache_dir,
                    overlay_dir,
                    memory_cache,
                )
            except Exception as exc:
                warnings.warn(
                    f"Mask2Former failed for {image_path}: {exc}"
                )
                missing_records.append(
                    {
                        **semantic_base_metadata(entry, response_family),
                        "generated_path": str(image_path),
                        "reason": f"segmentation_failed: {exc}",
                    }
                )
                continue

            cache_path = clean_text(
                segmentation_result["grouped_cache_path"]
            )
            segmentation_index[cache_path] = {
                "generated_path": str(image_path),
                "raw_segmentation_cache_path": str(
                    segmentation_result["raw_cache_path"]
                ),
                "grouped_segmentation_cache_path": str(
                    segmentation_result["grouped_cache_path"]
                ),
                "newly_inferred": segmentation_result["newly_inferred"],
                "raw_labels_by_group": json.dumps(
                    segmentation_result["raw_labels_by_group"],
                    ensure_ascii=False,
                ),
            }

            response_map = entry["response_map"]
            metrics = semantic_region_response_metrics(
                response_map,
                segmentation_result["grouped"],
                segmentation_result["id_to_group"],
                response_family,
            )
            for metric in metrics:
                detailed_records.append(
                    {
                        **semantic_base_metadata(entry, response_family),
                        "generated_path": str(image_path),
                        "generated_path_candidate_count": n_candidates,
                        "generated_path_resolution_mode": resolution_mode,
                        "grouped_segmentation_cache_path": str(
                            segmentation_result["grouped_cache_path"]
                        ),
                        "raw_labels_in_group": json.dumps(
                            segmentation_result[
                                "raw_labels_by_group"
                            ].get(metric["semantic_group"], []),
                            ensure_ascii=False,
                        ),
                        **metric,
                    }
                )

    progress.close()

    detailed = pd.DataFrame(detailed_records)
    missing = pd.DataFrame(missing_records)
    segmentation_df = pd.DataFrame(segmentation_index.values())

    save_csv(
        detailed,
        semantic_root / "semantic_response_detailed.csv",
    )
    save_csv(
        missing,
        semantic_root / "semantic_response_missing_or_failed.csv",
    )
    save_csv(
        segmentation_df,
        semantic_root / "segmentation_index.csv",
    )

    sample_summary = build_semantic_sample_summary(detailed)
    checkpoint_summary = build_semantic_checkpoint_summary(sample_summary)
    level_summary = build_semantic_checkpoint_summary(
        sample_summary,
        extra_groups=("target_level",),
    )
    dimension_summary = build_semantic_checkpoint_summary(
        sample_summary,
        extra_groups=("target_dim",),
    )
    dimension_level_summary = build_semantic_checkpoint_summary(
        sample_summary,
        extra_groups=("target_dim", "target_level"),
    )

    save_csv(
        sample_summary,
        semantic_root / "semantic_response_sample_summary.csv",
    )
    save_csv(
        checkpoint_summary,
        semantic_root / "semantic_response_ckpt_summary.csv",
    )
    save_csv(
        level_summary,
        semantic_root / "semantic_response_ckpt_level_summary.csv",
    )
    save_csv(
        dimension_summary,
        semantic_root / "semantic_response_ckpt_dimension_summary.csv",
    )
    save_csv(
        dimension_level_summary,
        semantic_root
        / "semantic_response_ckpt_dimension_level_summary.csv",
    )

    if not checkpoint_summary.empty:
        pairs = checkpoint_summary[
            ["experiment", "ckpt_tag"]
        ].drop_duplicates()
        for _, pair in pairs.iterrows():
            experiment = clean_text(pair["experiment"])
            ckpt_tag = clean_text(pair["ckpt_tag"])
            current_dir = ensure_dir(
                by_ckpt_dir
                / sanitize_filename(experiment)
                / sanitize_filename(ckpt_tag)
            )

            ckpt_sub = checkpoint_summary[
                (checkpoint_summary["experiment"].astype(str) == experiment)
                & (checkpoint_summary["ckpt_tag"].astype(str) == ckpt_tag)
            ].copy()
            level_sub = level_summary[
                (level_summary["experiment"].astype(str) == experiment)
                & (level_summary["ckpt_tag"].astype(str) == ckpt_tag)
            ].copy()
            dim_sub = dimension_summary[
                (dimension_summary["experiment"].astype(str) == experiment)
                & (dimension_summary["ckpt_tag"].astype(str) == ckpt_tag)
            ].copy()
            dim_level_sub = dimension_level_summary[
                (
                    dimension_level_summary["experiment"].astype(str)
                    == experiment
                )
                & (
                    dimension_level_summary["ckpt_tag"].astype(str)
                    == ckpt_tag
                )
            ].copy()

            save_csv(
                ckpt_sub,
                current_dir / "semantic_response_summary.csv",
            )
            save_csv(
                level_sub,
                current_dir / "semantic_response_by_level.csv",
            )
            save_csv(
                dim_sub,
                current_dir / "semantic_response_by_dimension.csv",
            )
            save_csv(
                dim_level_sub,
                current_dir
                / "semantic_response_by_dimension_and_level.csv",
            )

            # Control-analysis figure: one chart per experiment / checkpoint.
            save_semantic_polar_chart(
                level_summary,
                experiment,
                ckpt_tag,
                "control_analysis",
                current_dir / "semantic_response_radial_control_analysis.png",
            )

            # Cross-attention figure: one chart per experiment / checkpoint / dimension.
            cross_dims = [
                str(x)
                for x in sorted(
                    dim_level_sub[
                        dim_level_sub["response_family"].astype(str)
                        == "cross_attention"
                    ]["target_dim"].dropna().astype(str).unique().tolist()
                )
            ]
            for target_dim in cross_dims:
                save_semantic_polar_chart(
                    dim_level_sub,
                    experiment,
                    ckpt_tag,
                    "cross_attention",
                    current_dir
                    / f"semantic_response_radial_cross_attention_{sanitize_filename(target_dim)}.png",
                    target_dim=target_dim,
                )

    semantic_config = {
        "input_root": str(INPUT_ROOT),
        "output_root": str(semantic_root),
        "model_name": MASK2FORMER_MODEL_NAME,
        "device": str(runtime[4]),
        "generated_image_path_columns": GENERATED_IMAGE_PATH_COLUMNS,
        "main_timestep_only": MASK2FORMER_MAIN_TIMESTEP_ONLY,
        "main_timestep_fraction": MAIN_TIMESTEP_FRACTION,
        "timestep_tolerance": TIMESTEP_TOLERANCE,
        "attention_selection": "target phrase with target-token fallback",
        "attention_layer_groups": MASK2FORMER_ATTENTION_LAYER_GROUPS,
        "control_analysis_types": MASK2FORMER_CONTROL_ANALYSIS_TYPES,
        "control_block_names": MASK2FORMER_CONTROL_BLOCK_NAMES,
        "minimum_pixel_share_for_chart": (
            MASK2FORMER_MIN_PIXEL_SHARE_FOR_CHART
        ),
        "minimum_samples_for_chart": MASK2FORMER_MIN_SAMPLES_FOR_CHART,
        "radial_max_enrichment": MASK2FORMER_RADIAL_MAX_ENRICHMENT,
    }
    (semantic_root / "semantic_analysis_config.json").write_text(
        json.dumps(semantic_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[MASK2FORMER SAVE] detailed={len(detailed)}, "
        f"sample_summary={len(sample_summary)}, "
        f"checkpoint_summary={len(checkpoint_summary)}, "
        f"missing={len(missing)}"
    )

    torch_module = runtime[0]
    del runtime
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()

    return {
        "detailed": detailed,
        "sample_summary": sample_summary,
        "checkpoint_summary": checkpoint_summary,
        "level_summary": level_summary,
        "dimension_summary": dimension_summary,
        "dimension_level_summary": dimension_level_summary,
        "missing": missing,
        "segmentation_index": segmentation_df,
    }


# ============================================================
# 9. Main
# ============================================================


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {path} ({len(df)} rows)")


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    loaded = load_run(RUN_NAME, INPUT_ROOT)
    if loaded is None:
        raise RuntimeError(
            f"Input inference folder does not exist: {INPUT_ROOT}"
        )

    # Keep the original downstream table-based logic unchanged, but the list
    # now contains only the single folder selected in INPUT_ROOT.
    tables: List[RunTables] = [loaded]

    view_lookup = build_canonical_view_hint_lookup(tables)
    print(f"[VIEW HINT] canonical masks available for {len(view_lookup)} run/image pairs")

    attention_raw = pd.concat(
        [t.attention for t in tables if not t.attention.empty],
        ignore_index=True,
        sort=False,
    ) if any(not t.attention.empty for t in tables) else pd.DataFrame()

    control_raw = pd.concat(
        [t.control for t in tables if not t.control.empty],
        ignore_index=True,
        sort=False,
    ) if any(not t.control.empty for t in tables) else pd.DataFrame()

    attention_prepared = prepare_attention_rows(attention_raw)
    control_prepared = prepare_control_rows(control_raw)
    print(
        f"[PREPARE] attention rows: {len(attention_raw)} -> {len(attention_prepared)}; "
        f"control rows: {len(control_raw)} -> {len(control_prepared)}; "
        f"analysis map size: {ANALYSIS_MAP_SIZE}x{ANALYSIS_MAP_SIZE}"
    )

    # Cross-attention.
    attention_metrics, attention_maps = compute_attention_map_metrics(
        attention_prepared,
        view_lookup,
    )
    cue_agg_df, cue_agg_maps = aggregate_cue_maps(
        attention_prepared,
        attention_maps,
    )
    attention_change = compute_attention_level_change(
        attention_prepared,
        attention_maps,
        cue_agg_df,
        cue_agg_maps,
    )
    target_lookup = build_target_map_lookup(attention_prepared, attention_maps)
    target_cue_alignment = compute_target_cue_alignment(
        target_lookup,
        cue_agg_df,
        cue_agg_maps,
    )

    # Control features.
    control_metrics, control_maps = compute_control_map_metrics(
        control_prepared,
        view_lookup,
    )
    control_stability = compute_control_level_stability(
        control_prepared,
        control_maps,
    )

    summary = build_main_timestep_summary(
        attention_metrics,
        attention_change,
        target_cue_alignment,
        control_metrics,
        control_stability,
    )

    save_csv(attention_metrics, OUTPUT_DIR / "cross_attention_map_metrics.csv")
    save_csv(cue_agg_df, OUTPUT_DIR / "cross_attention_cue_aggregate_metrics.csv")
    save_csv(attention_change, OUTPUT_DIR / "cross_attention_level_change.csv")
    save_csv(
        target_cue_alignment,
        OUTPUT_DIR / "cross_attention_target_cue_alignment.csv",
    )
    save_csv(control_metrics, OUTPUT_DIR / "control_map_metrics.csv")
    save_csv(control_stability, OUTPUT_DIR / "control_level_stability.csv")
    save_csv(summary, OUTPUT_DIR / "mechanism_summary_main_timestep.csv")

    # Save the canonical view-hint index for reproducibility.
    view_rows = [
        {"run_name": run, "image_id": image_id, "view_hint_path": str(path)}
        for (run, image_id), path in sorted(view_lookup.items())
    ]
    save_csv(pd.DataFrame(view_rows), OUTPUT_DIR / "canonical_view_hint_index.csv")

    if SAVE_QUALITATIVE_PANELS:
        panel_dir = ensure_dir(OUTPUT_DIR / "qualitative_panels")
        build_attention_panels(attention_prepared, attention_maps, panel_dir)
        build_control_panels(control_prepared, control_maps, panel_dir)
        print(f"[SAVE] qualitative panels: {panel_dir}")

    config_payload = {
        "input_root": str(INPUT_ROOT),
        "run_name": RUN_NAME,
        "output_base_dir": str(OUTPUT_BASE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "enable_mask2former_analysis": ENABLE_MASK2FORMER_ANALYSIS,
        "mask2former_model_name": MASK2FORMER_MODEL_NAME,
        "main_timestep_fraction": MAIN_TIMESTEP_FRACTION,
        "timestep_tolerance": TIMESTEP_TOLERANCE,
        "attention_units": ATTENTION_UNITS,
        "attention_layer_groups": ATTENTION_LAYER_GROUPS,
        "analysis_map_size": ANALYSIS_MAP_SIZE,
        "attention_phrases_only": ATTENTION_PHRASES_ONLY,
        "semantic_mask_roots": {k: str(v) for k, v in SEMANTIC_MASK_ROOTS.items()},
        "semantic_classes": SEMANTIC_CLASSES,
        "control_analysis_types": CONTROL_ANALYSIS_TYPES,
        "panel_control_analysis_type": PANEL_CONTROL_ANALYSIS_TYPE,
        "level_order": LEVEL_ORDER,
        "experiment_order": EXPERIMENT_ORDER,
    }
    (OUTPUT_DIR / "analysis_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Run the newly added semantic overlay analysis after all original outputs
    # have already been saved. Therefore, a Mask2Former dependency/model error
    # does not discard the original post-hoc analysis results.
    if ENABLE_MASK2FORMER_ANALYSIS:
        run_mask2former_semantic_analysis(
            sweep_df=loaded.sweep,
            attention_df=attention_prepared,
            attention_maps=attention_maps,
            control_df=control_prepared,
            control_maps=control_maps,
        )

    print("\nMechanism analysis completed.")


if __name__ == "__main__":
    main()

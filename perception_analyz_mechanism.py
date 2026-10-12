# -*- coding: utf-8 -*-
"""
Post-hoc analysis for the cross-attention and ControlNet feature-map outputs
produced by the current single-dimensional and multi-dimensional inference
scripts.

The script does not re-run inference. It recursively scans existing inference
outputs, deduplicates per-sample and combined metadata, reads saved .npy maps,
and produces:

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

The code supports both directory structures:
  .../sweep/<experiment>/<checkpoint>/...
  .../sweep/<experiment>/<checkpoint>/cross_attention/...
  .../sweep/<experiment>/<checkpoint>/control_analysis/...

No argparse is used. Edit only the USER SETTINGS section.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# 0. USER SETTINGS
# ============================================================

# Add one or both runs. Non-existing roots are skipped with a warning.
INFERENCE_RUNS: Dict[str, Path] = {
    "single": Path(
        "/home/yuebing/GenAI_density_urbanization/urban_data/"
        "inference_output_singledim_ablation"
    ),
    "multi": Path(
        "/home/yuebing/GenAI_density_urbanization/urban_data/"
        "inference_outputs_multidim_ablation"
    ),
}

OUTPUT_DIR = Path(
    "/home/yuebing/GenAI_density_urbanization/urban_data/"
    "mechanism_analysis_single_multi"
)

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
# 8. Main
# ============================================================


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {path} ({len(df)} rows)")


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    tables: List[RunTables] = []
    for run_name, root in INFERENCE_RUNS.items():
        loaded = load_run(run_name, Path(root))
        if loaded is not None:
            tables.append(loaded)

    if not tables:
        raise RuntimeError("No valid inference output roots were found.")

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
        "inference_runs": {k: str(v) for k, v in INFERENCE_RUNS.items()},
        "output_dir": str(OUTPUT_DIR),
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

    print("\nMechanism analysis completed.")


if __name__ == "__main__":
    main()

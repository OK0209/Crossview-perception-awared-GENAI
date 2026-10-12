# -*- coding: utf-8 -*-
"""
Compare prompt-control and hint-control mechanisms saved by
perception_inference_multi_dim_cross.py.

Important interpretation:
- Cross-attention maps are token-conditioned probability/spatial response maps.
- ControlNet residual and delta maps are feature perturbation/injection strength.
- Their raw values have different units. This script always normalizes maps to
  0-1 and compares spatial distribution, region response, overlap, and time
  dynamics at the same timestep_fraction / DDIM step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image


# Adjust only this block if your output root changes.
BASE_DATA_DIR = Path("/home/yuebing/GenAI_density_urbanization/urban_data")
OUTPUT_ROOT = BASE_DATA_DIR / "inference_outputs_multidim_ablation"
SWEEP_ROOT = OUTPUT_ROOT / "sweep"
IMAGE_RESOLUTION = 512
TOPK_FRACTION = 0.10
SALIENT_QUANTILE = 0.90
PREFER_PHRASE_ATTENTION = True
PHRASE_PREFIXES = ("target_score_phrase:", "target_level_phrase:", "cue_phrase:")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_map(path: str, size: int = IMAGE_RESOLUTION) -> Optional[np.ndarray]:
    if not path or not Path(path).exists():
        return None
    if str(path).lower().endswith(".npy"):
        arr = np.load(path).astype(np.float32)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        arr = img.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_CUBIC)
    return normalize01(arr)


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn + 1e-8:
        return np.zeros(arr.shape, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    x = pd.Series(a.reshape(-1)).rank(method="average").to_numpy()
    y = pd.Series(b.reshape(-1)).rank(method="average").to_numpy()
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def topk_mask(a: np.ndarray, frac: float = TOPK_FRACTION) -> np.ndarray:
    flat = a.reshape(-1)
    k = max(1, int(round(flat.size * frac)))
    thr = np.partition(flat, flat.size - k)[flat.size - k]
    return a >= thr


def salient_mask(a: np.ndarray, quantile: float = SALIENT_QUANTILE) -> np.ndarray:
    return a >= float(np.quantile(a, quantile))


def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union > 0 else 0.0


def topk_overlap(a: np.ndarray, b: np.ndarray, frac: float = TOPK_FRACTION) -> float:
    ma = topk_mask(a, frac)
    mb = topk_mask(b, frac)
    denom = max(1, ma.sum())
    return float(np.logical_and(ma, mb).sum() / denom)


def read_rgb(path: str, size: int = IMAGE_RESOLUTION) -> Optional[np.ndarray]:
    if not path or not Path(path).exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def view_hint_masks(view_hint_path: str, size: int = IMAGE_RESOLUTION) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = read_rgb(view_hint_path, size)
    if img is None:
        z = np.zeros((size, size), dtype=bool)
        return z, z, ~z
    arr = img.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    nonwhite = np.min(255.0 - arr, axis=2) > 10
    road = nonwhite & (r > b + 25) & (g > b + 5)
    building = nonwhite & (b > r + 20) & (b > g + 10)
    background = ~(road | building)
    return road, building, background


def region_means(prefix: str, heat: np.ndarray, view_hint_path: str) -> Dict[str, float]:
    road, building, background = view_hint_masks(view_hint_path)

    def mm(mask):
        return float(heat[mask].mean()) if mask.any() else 0.0

    return {
        f"{prefix}_mean_inside_road_mask": mm(road),
        f"{prefix}_mean_inside_building_mask": mm(building),
        f"{prefix}_mean_inside_background_mask": mm(background),
    }


def heat_rgb(a: np.ndarray) -> np.ndarray:
    u8 = np.clip(normalize01(a) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def save_side_by_side(out_path: Path, panels: List[Tuple[str, np.ndarray]]) -> None:
    ensure_dir(out_path.parent)
    rendered = []
    for title, img in panels:
        if img.ndim == 2:
            img = heat_rgb(img)
        canvas = img.copy()
        cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        rendered.append(canvas)
    Image.fromarray(np.concatenate(rendered, axis=1)).save(out_path)


def save_overlap(out_path: Path, cross: np.ndarray, residual: np.ndarray, view_hint_path: str = "") -> None:
    ensure_dir(out_path.parent)
    cm = salient_mask(cross)
    rm = salient_mask(residual)
    base = read_rgb(view_hint_path) if view_hint_path else None
    if base is None:
        base = np.full((IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3), 255, dtype=np.uint8)
    overlay = base.astype(np.float32) * 0.55
    overlay[cm] = overlay[cm] * 0.35 + np.array([255, 40, 40]) * 0.65
    overlay[rm] = overlay[rm] * 0.35 + np.array([40, 120, 255]) * 0.65
    both = cm & rm
    overlay[both] = np.array([255, 230, 40])
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_path)


def discover_metadata() -> List[Tuple[Path, Path, Path]]:
    pairs = []
    for out_base in SWEEP_ROOT.glob("*/*"):
        if not out_base.is_dir():
            continue
        ca = out_base / "cross_attention_metadata.csv"
        ctrl = out_base / "control_analysis_metadata.csv"
        if ca.exists() and ctrl.exists():
            pairs.append((out_base, ca, ctrl))
    return pairs


def compare_one_out_base(out_base: Path, ca_path: Path, ctrl_path: Path) -> pd.DataFrame:
    ca = pd.read_csv(ca_path)
    ctrl = pd.read_csv(ctrl_path)
    if ca.empty or ctrl.empty:
        return pd.DataFrame()

    if PREFER_PHRASE_ATTENTION and "attention_unit" in ca.columns and "matched_keyword" in ca.columns:
        phrase_mask = ca["attention_unit"].astype(str).eq("phrase_average")
        phrase_token_mask = ca["matched_keyword"].astype(str).str.startswith(PHRASE_PREFIXES)
        if phrase_mask.any():
            ca = ca[phrase_mask | ~phrase_token_mask].copy()

    ctrl_res = ctrl[ctrl["analysis_type"].astype(str) == "control_residual"].copy()
    ctrl_delta = ctrl[ctrl["analysis_type"].astype(str) == "control_delta_middle"].copy()

    rows = []
    for _, arow in ca.iterrows():
        keys = (
            (ctrl_res["image_id"].astype(str) == str(arow["image_id"]))
            & (ctrl_res["target_dim"].astype(str) == str(arow["target_dim"]))
            & (ctrl_res["target_level"].astype(str) == str(arow["target_level"]))
            & (ctrl_res["ddim_step_index"].astype(int) == int(arow["ddim_step_index"]))
            & (np.isclose(ctrl_res["timestep_fraction"].astype(float), float(arow["timestep_fraction"])))
        )
        candidates = ctrl_res[keys]
        if candidates.empty:
            continue

        cross = load_map(str(arow.get("attention_npy_path", "")))
        if cross is None:
            continue

        dkeys = (
            (ctrl_delta["image_id"].astype(str) == str(arow["image_id"]))
            & (ctrl_delta["target_dim"].astype(str) == str(arow["target_dim"]))
            & (ctrl_delta["target_level"].astype(str) == str(arow["target_level"]))
            & (ctrl_delta["ddim_step_index"].astype(int) == int(arow["ddim_step_index"]))
        )
        delta_row = ctrl_delta[dkeys].head(1)
        delta = load_map(str(delta_row.iloc[0]["control_map_npy_path"])) if len(delta_row) else None

        for _, rrow in candidates.iterrows():
            residual = load_map(str(rrow.get("control_map_npy_path", "")))
            if residual is None:
                continue
            view_hint_path = str(rrow.get("view_hint_path", arow.get("view_hint_path", "")))
            generated_path = str(arow.get("generated_path", rrow.get("generated_path", "")))
            exp = str(arow["experiment"])
            ckpt = str(arow["ckpt_tag"])
            target_dim = str(arow["target_dim"])
            image_id = str(arow["image_id"])
            level = str(arow["target_level"])
            frac = float(arow["timestep_fraction"])
            frac_tag = f"{frac:.2f}".replace(".", "p")
            comp_dir = ensure_dir(out_base / "comparison" / target_dim / image_id / level)

            generated_img = read_rgb(generated_path)
            if generated_img is None:
                generated_img = heat_rgb(cross)
            side_path = comp_dir / f"cross_vs_residual_side_by_side_frac_{frac_tag}_block_{int(rrow['block_index']):02d}_token_{int(arow['token_index']):02d}.png"
            save_side_by_side(side_path, [("cross", cross), ("residual", residual), ("generated", generated_img)])

            overlap_path = comp_dir / f"cross_residual_overlap_frac_{frac_tag}_block_{int(rrow['block_index']):02d}_token_{int(arow['token_index']):02d}.png"
            save_overlap(overlap_path, cross, residual, view_hint_path)

            delta_side_path = ""
            delta_overlap = 0.0
            if delta is not None:
                delta_side = comp_dir / f"cross_vs_delta_side_by_side_frac_{frac_tag}_token_{int(arow['token_index']):02d}.png"
                view_img = read_rgb(view_hint_path)
                if view_img is None:
                    view_img = heat_rgb(delta)
                save_side_by_side(delta_side, [("cross", cross), ("delta", delta), ("view_hint", view_img)])
                delta_side_path = str(delta_side)
                delta_overlap = topk_overlap(cross, delta)

            metrics = {
                "pearson_correlation": pearson(cross, residual),
                "spearman_correlation": spearman(cross, residual),
                "cosine_similarity": cosine_similarity(cross, residual),
                "topk_overlap_ratio": topk_overlap(cross, residual),
                "iou_thresholded_salient_regions": mask_iou(salient_mask(cross), salient_mask(residual)),
                "overlap_high_attention_high_residual": topk_overlap(cross, residual),
                "overlap_high_attention_high_delta": delta_overlap,
            }
            metrics.update(region_means("attention", cross, view_hint_path))
            metrics.update(region_means("residual", residual, view_hint_path))
            if delta is not None:
                metrics.update(region_means("delta", delta, view_hint_path))

            row = {
                "experiment": exp,
                "ckpt_tag": ckpt,
                "image_id": image_id,
                "target_dim": target_dim,
                "target_level": level,
                "timestep_fraction": frac,
                "ddim_step_index": int(arow["ddim_step_index"]),
                "original_timestep_value": int(arow["original_timestep_value"]),
                "total_ddim_steps": int(arow["total_ddim_steps"]),
                "layer_group": str(arow.get("attention_layer_group", "")),
                "block": str(rrow.get("block_name", "")),
                "attention_unit": str(arow.get("attention_unit", "token")),
                "token_text": str(arow.get("token_text", "")),
                "token_index": int(arow.get("token_index", -1)),
                "cross_map_path": str(arow.get("attention_npy_path", "")),
                "residual_map_path": str(rrow.get("control_map_npy_path", "")),
                "delta_map_path": str(delta_row.iloc[0]["control_map_npy_path"]) if len(delta_row) else "",
                "view_hint_path": view_hint_path,
                "generated_path": generated_path,
                "side_by_side_path": str(side_path),
                "overlap_path": str(overlap_path),
                "delta_side_by_side_path": delta_side_path,
                **metrics,
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(out_base / "comparison" / "comparison_metrics.csv", index=False, encoding="utf-8-sig")
        meta = {
            "note": "Maps are normalized to 0-1 before comparison; raw cross-attention and residual values are not directly comparable.",
            "topk_fraction": TOPK_FRACTION,
            "salient_quantile": SALIENT_QUANTILE,
            "rows": int(len(out)),
        }
        (out_base / "comparison" / "comparison_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def write_overall_summary(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    group_cols = ["experiment", "target_dim", "target_level", "timestep_fraction"]
    metrics = [
        "pearson_correlation",
        "spearman_correlation",
        "cosine_similarity",
        "topk_overlap_ratio",
        "iou_thresholded_salient_regions",
        "attention_mean_inside_road_mask",
        "attention_mean_inside_building_mask",
        "residual_mean_inside_road_mask",
        "residual_mean_inside_building_mask",
    ]
    summary = df.groupby(group_cols)[metrics].mean().reset_index()
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dir(SWEEP_ROOT)
    all_rows = []
    for out_base, ca_path, ctrl_path in discover_metadata():
        print(f"[COMPARE] {out_base}")
        part = compare_one_out_base(out_base, ca_path, ctrl_path)
        if not part.empty:
            all_rows.append(part)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        summary_path = SWEEP_ROOT / "summary_cross_control_comparison.csv"
        combined.to_csv(summary_path, index=False, encoding="utf-8-sig")
        write_overall_summary(combined, SWEEP_ROOT / "overall_cross_control_summary.csv")
        print(f"[DONE] comparison summary saved: {summary_path}")
    else:
        print("[WARN] No comparable cross/control metadata pairs found.")


if __name__ == "__main__":
    main()

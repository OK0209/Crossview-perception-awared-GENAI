# -*- coding: utf-8 -*-
"""
Build montage panels from mechanism-analysis outputs produced by
perception_inference_multi_dim_cross.py.

This script is post-processing only:
- It does not load Stable Diffusion / ControlNet.
- It does not rerun inference.
- It only reads existing metadata/images and arranges overlay files into panels.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 0. USER CONFIGURATION
# ============================================================

OUTPUT_ROOT = Path("/home/yuebing/GenAI_density_urbanization/urban_data/inference_outputs_multidim_ablation")
IMAGE_DIR = Path("/home/yuebing/GenAI_density_urbanization/urban_data/images_new")

EXPERIMENT_ORDER = ["baseline", "view_only", "view_level_only", "full"]
DIMENSIONS = ["beautiful", "safety", "wealthy", "boring", "depressing", "lively"]
LEVEL_ORDER = ["low", "medium", "high"]

CROSS_ATTN_GROUPS = ["up", "mid", "down", "control"]
CROSS_ATTENTION_UNIT = "phrase_average"
CROSS_PANEL_COLUMNS = ["GT", "Generated", "early", "middle", "late"]

CONTROL_BLOCK_ROWS = [0,  6, 12]
CONTROL_PANEL_COLUMNS = ["GT", "Generated", "early", "middle", "late"]
CONTROL_OVERLAY_COLUMN = "control_overlay_path"
# Optional alternative:
# CONTROL_OVERLAY_COLUMN = "control_overlay_view_hint_path"

MAKE_EXTRA_DELTA_PANEL = False
ALLOW_HEATMAP_FALLBACK = False

PANEL_ROOT = OUTPUT_ROOT / "mechanism_panels"

CELL_SIZE = 256
LABEL_HEIGHT = 36
ROW_LABEL_WIDTH = 120
TITLE_HEIGHT = 70
GRID_LINE_WIDTH = 2
BACKGROUND_COLOR = (255, 255, 255)
MISSING_CELL_COLOR = (245, 245, 245)

OVERWRITE = True


# ============================================================
# 1. Utilities
# ============================================================

SWEEP_ROOT = OUTPUT_ROOT / "sweep"
TIMESTEP_LABELS = ["early", "middle", "late"]
SUMMARY_ROWS: List[Dict] = []
CELL_ROWS: List[Dict] = []


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_file(root_dir: Path, image_id_or_name: str) -> Optional[Path]:
    root = Path(root_dir)
    name = str(image_id_or_name)
    direct = root / name
    if direct.exists() and direct.is_file():
        return direct

    stem = Path(name).stem
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        p = root / f"{stem}{ext}"
        if p.exists() and p.is_file():
            return p
    return None


def sanitize_filename(text: object, max_len: int = 120) -> str:
    s = str(text)
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    s = s.strip("._-")
    return (s[:max_len] if s else "none")


def get_font(size: int) -> ImageFont.ImageFont:
    for name in ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def load_and_resize_image(path: Optional[Path], cell_size: int = CELL_SIZE) -> Optional[Image.Image]:
    if path is None or not Path(path).exists():
        return None
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None

    w, h = img.size
    if w <= 0 or h <= 0:
        return None
    scale = min(cell_size / w, cell_size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (cell_size, cell_size), BACKGROUND_COLOR)
    canvas.paste(img, ((cell_size - nw) // 2, (cell_size - nh) // 2))
    return canvas


def make_missing_cell(cell_size: int = CELL_SIZE, label: str = "Missing") -> Image.Image:
    img = Image.new("RGB", (cell_size, cell_size), MISSING_CELL_COLOR)
    draw = ImageDraw.Draw(img)
    font = get_font(22)
    bbox = draw.textbbox((0, 0), label, font=font)
    x = (cell_size - (bbox[2] - bbox[0])) // 2
    y = (cell_size - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), label, fill=(120, 120, 120), font=font)
    return img


def draw_centered_text(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, font, fill=(0, 0, 0)) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - tw) // 2, y0 + (y1 - y0 - th) // 2), text, fill=fill, font=font)


def draw_panel(
    title: str,
    row_labels: List[str],
    col_labels: List[str],
    cell_paths: Dict[Tuple[str, str], Optional[Path]],
    output_path: Path,
) -> Tuple[int, int]:
    ensure_dir(output_path.parent)
    n_rows, n_cols = len(row_labels), len(col_labels)
    width = ROW_LABEL_WIDTH + n_cols * CELL_SIZE
    height = TITLE_HEIGHT + LABEL_HEIGHT + n_rows * CELL_SIZE

    panel = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(panel)
    title_font = get_font(24)
    label_font = get_font(18)
    row_font = get_font(17)

    draw_centered_text(draw, (0, 0, width, TITLE_HEIGHT), title, title_font)

    for c, col in enumerate(col_labels):
        x0 = ROW_LABEL_WIDTH + c * CELL_SIZE
        draw_centered_text(draw, (x0, TITLE_HEIGHT, x0 + CELL_SIZE, TITLE_HEIGHT + LABEL_HEIGHT), col, label_font)

    existing = 0
    missing = 0
    y_grid0 = TITLE_HEIGHT + LABEL_HEIGHT
    for r, row in enumerate(row_labels):
        y0 = y_grid0 + r * CELL_SIZE
        draw_centered_text(draw, (0, y0, ROW_LABEL_WIDTH, y0 + CELL_SIZE), row, row_font)
        for c, col in enumerate(col_labels):
            x0 = ROW_LABEL_WIDTH + c * CELL_SIZE
            src = cell_paths.get((row, col))
            img = load_and_resize_image(src, CELL_SIZE)
            if img is None:
                img = make_missing_cell(CELL_SIZE)
                missing += 1
            else:
                existing += 1
            panel.paste(img, (x0, y0))

    line_color = (215, 215, 215)
    for c in range(n_cols + 1):
        x = ROW_LABEL_WIDTH + c * CELL_SIZE
        draw.line((x, TITLE_HEIGHT, x, height), fill=line_color, width=GRID_LINE_WIDTH)
    for r in range(n_rows + 1):
        y = y_grid0 + r * CELL_SIZE
        draw.line((0, y, width, y), fill=line_color, width=GRID_LINE_WIDTH)
    draw.line((0, TITLE_HEIGHT, width, TITLE_HEIGHT), fill=line_color, width=GRID_LINE_WIDTH)

    if OVERWRITE or not output_path.exists():
        panel.save(output_path)
    return existing, missing


def derive_cross_attention_overlay_path(attention_png_path: object) -> Optional[Path]:
    if pd.isna(attention_png_path):
        return None
    p = Path(str(attention_png_path))
    if not p.name:
        return None
    if p.name.startswith("attn_"):
        return p.with_name("attn_overlay_" + p.name[len("attn_"):])
    return p.with_name("attn_overlay_" + p.name)


def infer_timestep_labels(metadata_df: pd.DataFrame) -> Dict[str, float]:
    if metadata_df.empty or "timestep_fraction" not in metadata_df.columns:
        return {}
    vals = sorted(float(x) for x in metadata_df["timestep_fraction"].dropna().unique())
    if not vals:
        return {}
    if len(vals) == 1:
        return {"early": vals[0], "middle": vals[0], "late": vals[0]}
    if len(vals) == 2:
        return {"early": vals[0], "middle": vals[0], "late": vals[1]}
    return {"early": vals[0], "middle": vals[len(vals) // 2], "late": vals[-1]}


def path_exists(path: Optional[Path]) -> bool:
    if path is None:
        return False
    s = str(path).strip()
    return bool(s) and Path(path).exists() and Path(path).is_file()


def select_best_cross_row(candidate_rows: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[Path], str]:
    if candidate_rows.empty:
        return None, None, "no matching metadata row"
    candidates = []
    for idx, row in candidate_rows.iterrows():
        overlay = derive_cross_attention_overlay_path(row.get("attention_png_path", ""))
        exists = path_exists(overlay)
        fallback = Path(str(row.get("attention_png_path", ""))) if ALLOW_HEATMAP_FALLBACK and not exists else overlay
        fallback_exists = path_exists(fallback)
        layers = int(row.get("attention_layers_aggregated", 0)) if not pd.isna(row.get("attention_layers_aggregated", np.nan)) else 0
        candidates.append((fallback_exists, exists, layers, idx, fallback))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    _, overlay_exists, _, idx, chosen_path = candidates[0]
    note = "ok" if path_exists(chosen_path) else "overlay missing"
    if chosen_path is not None and not overlay_exists and ALLOW_HEATMAP_FALLBACK and path_exists(chosen_path):
        note = "used heatmap fallback"
    return candidate_rows.loc[idx], chosen_path, note


def select_best_path_row(candidate_rows: pd.DataFrame, path_column: str) -> Tuple[Optional[pd.Series], Optional[Path], str]:
    if candidate_rows.empty:
        return None, None, "no matching metadata row"
    candidates = []
    for idx, row in candidate_rows.iterrows():
        p = Path(str(row.get(path_column, ""))) if not pd.isna(row.get(path_column, np.nan)) else None
        exists = path_exists(p)
        candidates.append((exists, idx, p))
    candidates.sort(key=lambda x: x[0], reverse=True)
    exists, idx, p = candidates[0]
    return candidate_rows.loc[idx], p, "ok" if exists else f"{path_column} missing"


def discover_sweep_folders() -> List[Tuple[str, str, Path]]:
    folders = []
    for exp in EXPERIMENT_ORDER:
        exp_dir = SWEEP_ROOT / exp
        if not exp_dir.exists():
            continue
        for ckpt_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir()]):
            if (ckpt_dir / "metadata_sweep.csv").exists():
                folders.append((exp, ckpt_dir.name, ckpt_dir))
    return folders


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return pd.DataFrame()


def normalize_level(x: object) -> str:
    return str(x).strip().lower().replace("_", " ").replace("-", " ")


def get_gt_path(meta_rows: pd.DataFrame, image_id: str) -> Optional[Path]:
    if not meta_rows.empty and "image_name_raw" in meta_rows.columns:
        name = str(meta_rows.iloc[0].get("image_name_raw", image_id))
        return find_file(IMAGE_DIR, name) or find_file(IMAGE_DIR, image_id)
    return find_file(IMAGE_DIR, image_id)


def get_generated_path(meta_rows: pd.DataFrame, level: str) -> Optional[Path]:
    if meta_rows.empty:
        return None
    level_rows = meta_rows[meta_rows["target_level"].astype(str).map(normalize_level) == level]
    if level_rows.empty:
        return None
    p = Path(str(level_rows.iloc[0].get("generated_path", "")))
    return p if p.exists() else p


def add_cell_index(
    panel_type: str,
    panel_path: Path,
    experiment: str,
    ckpt_tag: str,
    target_dim: str,
    image_id: str,
    level: str,
    phrase_group: str,
    phrase_attribute: str,
    attention_group: str,
    control_block: object,
    timestep_label: str,
    timestep_fraction: object,
    row_label: str,
    column_label: str,
    source_path: Optional[Path],
    note: str,
) -> None:
    CELL_ROWS.append({
        "panel_type": panel_type,
        "panel_path": str(panel_path),
        "experiment": experiment,
        "ckpt_tag": ckpt_tag,
        "target_dim": target_dim,
        "image_id": image_id,
        "level": level,
        "phrase_group": phrase_group,
        "phrase_attribute": phrase_attribute,
        "attention_group": attention_group,
        "control_block": control_block,
        "timestep_label": timestep_label,
        "timestep_fraction": timestep_fraction,
        "row_label": row_label,
        "column_label": column_label,
        "source_path": str(source_path) if source_path is not None else "",
        "source_exists": bool(path_exists(source_path)),
        "note": note,
    })


def add_panel_summary(
    panel_type: str,
    panel_path: Path,
    experiment: str,
    ckpt_tag: str,
    target_dim: str,
    image_id: str,
    level: str,
    phrase_group: str,
    phrase_attribute: str,
    attention_group: str,
    num_cells: int,
    num_existing_cells: int,
    num_missing_cells: int,
) -> None:
    SUMMARY_ROWS.append({
        "panel_type": panel_type,
        "panel_path": str(panel_path),
        "experiment": experiment,
        "ckpt_tag": ckpt_tag,
        "target_dim": target_dim,
        "image_id": image_id,
        "level": level,
        "phrase_group": phrase_group,
        "phrase_attribute": phrase_attribute,
        "attention_group": attention_group,
        "num_cells": num_cells,
        "num_existing_cells": num_existing_cells,
        "num_missing_cells": num_missing_cells,
    })


# ============================================================
# 2. Cross-attention panels
# ============================================================

def parse_phrase(matched_keyword: str) -> Tuple[str, str]:
    text = str(matched_keyword)
    if text.startswith("target_score_phrase:"):
        return "target_score_phrase", ""
    if text.startswith("target_level_phrase:"):
        return "target_level_phrase", ""
    if text.startswith("cue_phrase:"):
        parts = text.split(":", 3)
        return "cue_phrase", parts[3] if len(parts) >= 4 else ""
    return "", ""


def phrase_specs_for(exp: str, target_dim: str, ca_subset: pd.DataFrame) -> List[Tuple[str, str]]:
    if exp in ["baseline", "view_only"]:
        return [("target_score_phrase", "")]
    if exp == "view_level_only":
        return [("target_level_phrase", "")]
    if exp == "full":
        specs = [("target_level_phrase", "")]
        mk = ca_subset.get("matched_keyword", pd.Series(dtype=str)).astype(str)
        attrs = sorted({
            parse_phrase(x)[1]
            for x in mk
            if x.startswith(f"cue_phrase:{target_dim}:") and parse_phrase(x)[1]
        })
        specs.extend(("cue_phrase", attr) for attr in attrs)
        return specs
    return []


def expected_matched_keyword(phrase_group: str, target_dim: str, level: str, phrase_attribute: str) -> str:
    if phrase_group == "target_score_phrase":
        return f"target_score_phrase:{target_dim}:"
    if phrase_group == "target_level_phrase":
        return f"target_level_phrase:{target_dim}:{level}"
    if phrase_group == "cue_phrase":
        return f"cue_phrase:{target_dim}:{level}:{phrase_attribute}"
    return ""


def build_cross_attention_panels(exp: str, ckpt_tag: str, folder: Path, sweep_meta: pd.DataFrame, ca: pd.DataFrame) -> int:
    if sweep_meta.empty or ca.empty:
        return 0
    required = {"image_id", "target_dim", "target_level", "attention_layer_group", "matched_keyword", "attention_unit"}
    if not required.issubset(set(ca.columns)):
        return 0

    ca = ca[ca["attention_unit"].astype(str) == CROSS_ATTENTION_UNIT].copy()
    if ca.empty:
        return 0

    count = 0
    dims = [d for d in DIMENSIONS if d in set(ca["target_dim"].astype(str))]
    for target_dim in dims:
        dim_ca = ca[ca["target_dim"].astype(str) == target_dim]
        for image_id in sorted(dim_ca["image_id"].astype(str).unique()):
            image_ca = dim_ca[dim_ca["image_id"].astype(str) == image_id]
            meta_img_dim = sweep_meta[
                (sweep_meta["image_id"].astype(str) == image_id)
                & (sweep_meta["target_dim"].astype(str) == target_dim)
            ]
            gt_path = get_gt_path(meta_img_dim, image_id)
            phrase_specs = phrase_specs_for(exp, target_dim, image_ca)
            for phrase_group, phrase_attribute in phrase_specs:
                phrase_key = phrase_group if not phrase_attribute else f"{phrase_group}__{sanitize_filename(phrase_attribute)}"
                for attn_group in CROSS_ATTN_GROUPS:
                    group_ca = image_ca[image_ca["attention_layer_group"].astype(str) == attn_group]
                    if group_ca.empty:
                        continue
                    timestep_map = infer_timestep_labels(group_ca)
                    if not timestep_map:
                        continue

                    out_dir = ensure_dir(PANEL_ROOT / "cross_attention" / exp / ckpt_tag / target_dim / image_id / attn_group / phrase_key)
                    if phrase_group == "cue_phrase":
                        fname = f"cross_attn_panel__cue_phrase__{sanitize_filename(phrase_attribute)}__{attn_group}.png"
                    else:
                        fname = f"cross_attn_panel__{phrase_group}__{attn_group}.png"
                    panel_path = out_dir / fname
                    title = f"{exp} | {ckpt_tag} | {target_dim} | {image_id} | {phrase_group}:{phrase_attribute} | attn-{attn_group}"

                    cell_paths: Dict[Tuple[str, str], Optional[Path]] = {}
                    for level in LEVEL_ORDER:
                        gen_path = get_generated_path(meta_img_dim, level)
                        cell_paths[(level, "GT")] = gt_path
                        add_cell_index("cross_attention", panel_path, exp, ckpt_tag, target_dim, image_id, level,
                                       phrase_group, phrase_attribute, attn_group, "", "", "", level, "GT",
                                       gt_path, "ok" if path_exists(gt_path) else "GT missing")
                        cell_paths[(level, "Generated")] = gen_path
                        add_cell_index("cross_attention", panel_path, exp, ckpt_tag, target_dim, image_id, level,
                                       phrase_group, phrase_attribute, attn_group, "", "", "", level, "Generated",
                                       gen_path, "ok" if path_exists(gen_path) else "generated missing")

                        for tlabel in TIMESTEP_LABELS:
                            frac = timestep_map.get(tlabel, np.nan)
                            expected = expected_matched_keyword(phrase_group, target_dim, level, phrase_attribute)
                            level_ca = group_ca[
                                (group_ca["target_level"].astype(str).map(normalize_level) == level)
                                & (np.isclose(group_ca["timestep_fraction"].astype(float), float(frac)))
                            ]
                            if phrase_group == "target_score_phrase":
                                cand = level_ca[level_ca["matched_keyword"].astype(str).str.startswith(expected)]
                            else:
                                cand = level_ca[level_ca["matched_keyword"].astype(str) == expected]
                            chosen, src, note = select_best_cross_row(cand)
                            cell_paths[(level, tlabel)] = src
                            add_cell_index("cross_attention", panel_path, exp, ckpt_tag, target_dim, image_id, level,
                                           phrase_group, phrase_attribute, attn_group, "", tlabel, frac,
                                           level, tlabel, src, note)

                    existing, missing = draw_panel(title, LEVEL_ORDER, CROSS_PANEL_COLUMNS, cell_paths, panel_path)
                    add_panel_summary("cross_attention", panel_path, exp, ckpt_tag, target_dim, image_id, "",
                                      phrase_group, phrase_attribute, attn_group,
                                      len(LEVEL_ORDER) * len(CROSS_PANEL_COLUMNS), existing, missing)
                    count += 1
    return count


# ============================================================
# 3. ControlNet panels
# ============================================================

def build_control_panels(exp: str, ckpt_tag: str, folder: Path, sweep_meta: pd.DataFrame, ctrl: pd.DataFrame) -> int:
    if sweep_meta.empty or ctrl.empty:
        return 0
    required = {"image_id", "target_dim", "target_level", "analysis_type", "block_index", "timestep_fraction"}
    if not required.issubset(set(ctrl.columns)):
        return 0
    ctrl_res = ctrl[ctrl["analysis_type"].astype(str) == "control_residual"].copy()
    if ctrl_res.empty:
        return 0

    count = 0
    dims = [d for d in DIMENSIONS if d in set(ctrl_res["target_dim"].astype(str))]
    row_labels = [f"block{b}" for b in CONTROL_BLOCK_ROWS]
    for target_dim in dims:
        dim_ctrl = ctrl_res[ctrl_res["target_dim"].astype(str) == target_dim]
        for image_id in sorted(dim_ctrl["image_id"].astype(str).unique()):
            meta_img_dim = sweep_meta[
                (sweep_meta["image_id"].astype(str) == image_id)
                & (sweep_meta["target_dim"].astype(str) == target_dim)
            ]
            gt_path = get_gt_path(meta_img_dim, image_id)
            for level in LEVEL_ORDER:
                group_ctrl = dim_ctrl[
                    (dim_ctrl["image_id"].astype(str) == image_id)
                    & (dim_ctrl["target_level"].astype(str).map(normalize_level) == level)
                ]
                if group_ctrl.empty:
                    continue
                timestep_map = infer_timestep_labels(group_ctrl)
                if not timestep_map:
                    continue

                out_dir = ensure_dir(PANEL_ROOT / "control_analysis" / exp / ckpt_tag / target_dim / image_id / f"level_{level}")
                panel_path = out_dir / f"control_panel__blocks_0_3_6_9_12__level_{level}.png"
                title = f"{exp} | {ckpt_tag} | {target_dim} | {image_id} | level={level} | control blocks"
                gen_path = get_generated_path(meta_img_dim, level)
                cell_paths: Dict[Tuple[str, str], Optional[Path]] = {}

                for block_idx, row_label in zip(CONTROL_BLOCK_ROWS, row_labels):
                    cell_paths[(row_label, "GT")] = gt_path
                    add_cell_index("control_analysis", panel_path, exp, ckpt_tag, target_dim, image_id, level,
                                   "", "", "", block_idx, "", "", row_label, "GT",
                                   gt_path, "ok" if path_exists(gt_path) else "GT missing")
                    cell_paths[(row_label, "Generated")] = gen_path
                    add_cell_index("control_analysis", panel_path, exp, ckpt_tag, target_dim, image_id, level,
                                   "", "", "", block_idx, "", "", row_label, "Generated",
                                   gen_path, "ok" if path_exists(gen_path) else "generated missing")

                    for tlabel in TIMESTEP_LABELS:
                        frac = timestep_map.get(tlabel, np.nan)
                        cand = group_ctrl[
                            (group_ctrl["block_index"].astype(int) == int(block_idx))
                            & (np.isclose(group_ctrl["timestep_fraction"].astype(float), float(frac)))
                        ]
                        chosen, src, note = select_best_path_row(cand, CONTROL_OVERLAY_COLUMN)
                        cell_paths[(row_label, tlabel)] = src
                        add_cell_index("control_analysis", panel_path, exp, ckpt_tag, target_dim, image_id, level,
                                       "", "", "", block_idx, tlabel, frac, row_label, tlabel, src, note)

                existing, missing = draw_panel(title, row_labels, CONTROL_PANEL_COLUMNS, cell_paths, panel_path)
                add_panel_summary("control_analysis", panel_path, exp, ckpt_tag, target_dim, image_id, level,
                                  "", "", "", len(row_labels) * len(CONTROL_PANEL_COLUMNS), existing, missing)
                count += 1
    return count


# ============================================================
# 4. Main
# ============================================================

def main() -> None:
    ensure_dir(PANEL_ROOT)
    cross_count = 0
    control_count = 0

    folders = discover_sweep_folders()
    if not folders:
        print(f"[WARN] No sweep folders found under {SWEEP_ROOT}")

    for exp, ckpt_tag, folder in folders:
        print(f"[PANEL] {exp}/{ckpt_tag}")
        sweep_meta = read_csv_if_exists(folder / "metadata_sweep.csv")
        ca = read_csv_if_exists(folder / "cross_attention_metadata.csv")
        ctrl = read_csv_if_exists(folder / "control_analysis_metadata.csv")
        try:
            cross_count += build_cross_attention_panels(exp, ckpt_tag, folder, sweep_meta, ca)
        except Exception as e:
            print(f"[WARN] cross-attention panels failed for {folder}: {e}")
        try:
            control_count += build_control_panels(exp, ckpt_tag, folder, sweep_meta, ctrl)
        except Exception as e:
            print(f"[WARN] control panels failed for {folder}: {e}")

    summary_df = pd.DataFrame(SUMMARY_ROWS)
    cell_df = pd.DataFrame(CELL_ROWS)
    summary_df.to_csv(PANEL_ROOT / "panel_summary.csv", index=False, encoding="utf-8-sig")
    cell_df.to_csv(PANEL_ROOT / "panel_cell_index.csv", index=False, encoding="utf-8-sig")

    missing_cells = int((~cell_df["source_exists"].astype(bool)).sum()) if not cell_df.empty else 0
    print("\n========== Panel Build Summary ==========")
    print(f"cross-attention panels generated : {cross_count}")
    print(f"control-analysis panels generated: {control_count}")
    print(f"missing cells                    : {missing_cells}")
    print(f"output directory                 : {PANEL_ROOT}")
    print("=========================================\n")


if __name__ == "__main__":
    main()

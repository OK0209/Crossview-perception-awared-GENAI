# -*- coding: utf-8 -*-
"""
Build 3x6 generated-image sweep panels from existing sweep outputs.

Panel layout:
             Beautiful | Safety | Lively | Wealthy | Boring | Depressing
    Low
    Medium
    High

One panel is created for each:
    experiment × checkpoint × image_id

This script is a pure post-processing utility:
- It does NOT load the diffusion model.
- It does NOT rerun inference.
- It only reads metadata_sweep.csv and generated images.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


# ============================================================
# 0. USER SETTINGS
# ============================================================

OUTPUT_ROOT = Path(
    "/home/yuebing/GenAI_density_urbanization/urban_data/inference_outputs_multidim_ablation"
)

EXPERIMENT_ORDER = [
    "baseline",
    "view_only",
    "view_level_only",
    "full",
]

# 按照你附图中的列顺序排列
DIMENSIONS = [
    "beautiful",
    "safety",
    "lively",
    "wealthy",
    "boring",
    "depressing",
]

DIM_LABELS = {
    "beautiful": "Beautiful",
    "safety": "Safety",
    "lively": "Lively",
    "wealthy": "Wealthy",
    "boring": "Boring",
    "depressing": "Depressing",
}

LEVEL_ORDER = ["low", "medium", "high"]

LEVEL_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}

# 输出目录
PANEL_ROOT = OUTPUT_ROOT / "generated_sweep_panels"

# 图像尺寸设置
CELL_SIZE = 256
ROW_LABEL_WIDTH = 110
COL_LABEL_HEIGHT = 42
TITLE_HEIGHT = 70
GRID_LINE = 2

BACKGROUND_COLOR = (255, 255, 255)
HEADER_BG = (250, 250, 250)
MISSING_BG = (245, 245, 245)
GRID_COLOR = (210, 210, 210)
TEXT_COLOR = (20, 20, 20)

# 是否覆盖已有 panel
OVERWRITE = True

# 如果只想处理部分 image_id，可以填列表；None 表示处理全部
TARGET_IMAGE_IDS: Optional[List[str]] = None

# 如果只想处理部分 ckpt，可以填关键词，例如 ["epoch9", "best"]；None 表示处理全部
CKPT_KEYWORDS: Optional[List[str]] = None


# ============================================================
# 1. Basic utilities
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(text: str, max_len: int = 120) -> str:
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if text else "unknown"


def normalize_level(x) -> str:
    if pd.isna(x):
        return "medium"
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    if s in ["very low", "low"]:
        return "low"
    if s in ["medium", "mid"]:
        return "medium"
    if s in ["very high", "high"]:
        return "high"
    if s in ["0"]:
        return "low"
    if s in ["1"]:
        return "medium"
    if s in ["2"]:
        return "high"
    return s


def normalize_dim(x) -> str:
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ])

    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    try:
        return draw.textbbox((0, 0), text, font=font)
    except Exception:
        w, h = draw.textsize(text, font=font)
        return (0, 0, w, h)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int] = TEXT_COLOR,
) -> None:
    x0, y0, x1, y1 = box
    bbox = text_bbox(draw, text, font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = x0 + (x1 - x0 - tw) / 2
    y = y0 + (y1 - y0 - th) / 2
    draw.text((x, y), text, font=font, fill=fill)


def make_missing_cell(size: int = CELL_SIZE, label: str = "Missing") -> Image.Image:
    img = Image.new("RGB", (size, size), MISSING_BG)
    draw = ImageDraw.Draw(img)
    font = get_font(22)
    draw_centered_text(draw, (0, 0, size, size), label, font, fill=(120, 120, 120))
    return img


def load_image_square(path: Optional[Path], size: int = CELL_SIZE) -> Image.Image:
    if path is None or not Path(path).exists():
        return make_missing_cell(size)

    try:
        img = Image.open(path).convert("RGB")
        # 保持比例，居中裁剪成正方形
        img = ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS)
        return img
    except Exception:
        return make_missing_cell(size)


def path_exists(p: object) -> bool:
    if p is None or pd.isna(p):
        return False
    return Path(str(p)).exists()


def ckpt_allowed(ckpt_tag: str) -> bool:
    if CKPT_KEYWORDS is None:
        return True
    return any(k in ckpt_tag for k in CKPT_KEYWORDS)


# ============================================================
# 2. Metadata and path matching
# ============================================================

def find_sweep_dirs(output_root: Path) -> List[Tuple[str, str, Path]]:
    """
    Return a list of:
        (experiment, ckpt_tag, sweep_dir)

    Expected structure:
        OUTPUT_ROOT / sweep / experiment / ckpt_tag / metadata_sweep.csv
    """
    sweep_root = output_root / "sweep"
    found = []

    if not sweep_root.exists():
        raise FileNotFoundError(f"Sweep root not found: {sweep_root}")

    for exp_name in EXPERIMENT_ORDER:
        exp_dir = sweep_root / exp_name
        if not exp_dir.exists():
            print(f"[WARN] Missing experiment dir: {exp_dir}")
            continue

        for ckpt_dir in sorted(exp_dir.iterdir()):
            if not ckpt_dir.is_dir():
                continue
            ckpt_tag = ckpt_dir.name
            if not ckpt_allowed(ckpt_tag):
                continue

            meta_path = ckpt_dir / "metadata_sweep.csv"
            if meta_path.exists():
                found.append((exp_name, ckpt_tag, ckpt_dir))
            else:
                print(f"[WARN] metadata_sweep.csv not found: {meta_path}")

    return found


def prepare_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    required = ["image_id", "target_dim", "target_level", "generated_path"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"metadata_sweep.csv missing required columns: {missing}")

    df = meta.copy()
    df["_image_id"] = df["image_id"].astype(str)
    df["_target_dim"] = df["target_dim"].map(normalize_dim)
    df["_target_level"] = df["target_level"].map(normalize_level)
    return df


def fallback_generated_path(
    sweep_dir: Path,
    image_id: str,
    dim: str,
    level: str,
) -> Path:
    """
    Fallback to the expected path used by the inference script:
        sweep_dir / target_dim / image_id / f"{image_id}_{target_dim}_{level}.png"
    """
    level_tag = str(level).replace(" ", "_")
    return sweep_dir / dim / image_id / f"{image_id}_{dim}_{level_tag}.png"


def select_generated_path(
    df: pd.DataFrame,
    sweep_dir: Path,
    image_id: str,
    dim: str,
    level: str,
) -> Tuple[Optional[Path], str]:
    """
    Select one generated image for a given image × dimension × level.
    Prefer metadata generated_path if it exists.
    Otherwise use the deterministic fallback path.
    """
    sub = df[
        (df["_image_id"] == str(image_id))
        & (df["_target_dim"] == dim)
        & (df["_target_level"] == level)
    ].copy()

    if len(sub) == 0:
        fb = fallback_generated_path(sweep_dir, image_id, dim, level)
        if fb.exists():
            return fb, "fallback_path"
        return None, "missing_metadata_row"

    # Prefer rows whose generated_path exists.
    for _, row in sub.iterrows():
        p = Path(str(row["generated_path"]))
        if p.exists():
            return p, "metadata_generated_path"

    # Fallback if absolute paths changed or metadata path is stale.
    fb = fallback_generated_path(sweep_dir, image_id, dim, level)
    if fb.exists():
        return fb, "fallback_path"

    # Last: return the first metadata path even if missing, for logging.
    p0 = Path(str(sub.iloc[0]["generated_path"]))
    return p0 if p0.exists() else None, "missing_file"


# ============================================================
# 3. Panel drawing
# ============================================================

def draw_sweep_panel(
    cell_paths: Dict[Tuple[str, str], Optional[Path]],
    title: str,
    output_path: Path,
) -> Dict[str, int]:
    """
    cell_paths keys:
        (level, dim)
    """
    n_rows = len(LEVEL_ORDER)
    n_cols = len(DIMENSIONS)

    width = ROW_LABEL_WIDTH + n_cols * CELL_SIZE + (n_cols + 1) * GRID_LINE
    height = (
        TITLE_HEIGHT
        + COL_LABEL_HEIGHT
        + n_rows * CELL_SIZE
        + (n_rows + 1) * GRID_LINE
    )

    canvas = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    title_font = get_font(24, bold=True)
    header_font = get_font(18, bold=True)
    row_font = get_font(18, bold=True)

    # Title
    draw_centered_text(
        draw,
        (0, 0, width, TITLE_HEIGHT),
        title,
        title_font,
        fill=TEXT_COLOR,
    )

    grid_top = TITLE_HEIGHT + COL_LABEL_HEIGHT

    # Header background
    draw.rectangle(
        [0, TITLE_HEIGHT, width, grid_top],
        fill=HEADER_BG,
    )

    # Column labels
    for c, dim in enumerate(DIMENSIONS):
        x0 = ROW_LABEL_WIDTH + GRID_LINE + c * (CELL_SIZE + GRID_LINE)
        x1 = x0 + CELL_SIZE
        draw_centered_text(
            draw,
            (x0, TITLE_HEIGHT, x1, grid_top),
            DIM_LABELS.get(dim, dim.title()),
            header_font,
        )

    # Row labels and cells
    existing = 0
    missing = 0

    for r, level in enumerate(LEVEL_ORDER):
        y0 = grid_top + GRID_LINE + r * (CELL_SIZE + GRID_LINE)
        y1 = y0 + CELL_SIZE

        # Row label
        draw.rectangle([0, y0, ROW_LABEL_WIDTH, y1], fill=HEADER_BG)
        draw_centered_text(
            draw,
            (0, y0, ROW_LABEL_WIDTH, y1),
            LEVEL_LABELS.get(level, level.title()),
            row_font,
        )

        for c, dim in enumerate(DIMENSIONS):
            x0 = ROW_LABEL_WIDTH + GRID_LINE + c * (CELL_SIZE + GRID_LINE)
            x1 = x0 + CELL_SIZE

            p = cell_paths.get((level, dim))
            if p is not None and Path(p).exists():
                img = load_image_square(p, CELL_SIZE)
                existing += 1
            else:
                img = make_missing_cell(CELL_SIZE)
                missing += 1

            canvas.paste(img, (x0, y0))

    # Grid lines
    for c in range(n_cols + 1):
        x = ROW_LABEL_WIDTH + c * (CELL_SIZE + GRID_LINE)
        draw.rectangle([x, grid_top, x + GRID_LINE - 1, height], fill=GRID_COLOR)

    for r in range(n_rows + 1):
        y = grid_top + r * (CELL_SIZE + GRID_LINE)
        draw.rectangle([0, y, width, y + GRID_LINE - 1], fill=GRID_COLOR)

    # Outer line around row-label/header split
    draw.rectangle([ROW_LABEL_WIDTH - GRID_LINE, TITLE_HEIGHT, ROW_LABEL_WIDTH, height], fill=GRID_COLOR)

    ensure_dir(output_path.parent)
    canvas.save(output_path, quality=95)

    return {
        "existing": existing,
        "missing": missing,
        "total": n_rows * n_cols,
    }


# ============================================================
# 4. Main panel generation
# ============================================================

def build_panels_for_sweep_dir(
    experiment: str,
    ckpt_tag: str,
    sweep_dir: Path,
) -> Tuple[List[Dict], List[Dict]]:
    meta_path = sweep_dir / "metadata_sweep.csv"
    meta = pd.read_csv(meta_path)
    df = prepare_metadata(meta)

    image_ids = sorted(df["_image_id"].dropna().astype(str).unique().tolist())
    if TARGET_IMAGE_IDS is not None:
        target_set = set(str(x) for x in TARGET_IMAGE_IDS)
        image_ids = [x for x in image_ids if x in target_set]

    summary_rows = []
    cell_rows = []

    print(f"[PANEL] {experiment}/{ckpt_tag}: {len(image_ids)} images")

    for image_id in image_ids:
        cell_paths: Dict[Tuple[str, str], Optional[Path]] = {}
        cell_notes: Dict[Tuple[str, str], str] = {}

        for level in LEVEL_ORDER:
            for dim in DIMENSIONS:
                p, note = select_generated_path(
                    df=df,
                    sweep_dir=sweep_dir,
                    image_id=image_id,
                    dim=dim,
                    level=level,
                )
                cell_paths[(level, dim)] = p
                cell_notes[(level, dim)] = note

        title = f"{experiment} | {ckpt_tag} | {image_id}"
        out_path = (
            PANEL_ROOT
            / "by_experiment"
            / experiment
            / ckpt_tag
            / f"{sanitize_filename(image_id)}__sweep_grid_3x6.png"
        )

        if out_path.exists() and not OVERWRITE:
            continue

        stats = draw_sweep_panel(
            cell_paths=cell_paths,
            title=title,
            output_path=out_path,
        )

        summary_rows.append({
            "panel_type": "generated_sweep_3x6",
            "experiment": experiment,
            "ckpt_tag": ckpt_tag,
            "image_id": image_id,
            "panel_path": str(out_path),
            "num_cells": stats["total"],
            "num_existing_cells": stats["existing"],
            "num_missing_cells": stats["missing"],
        })

        for r, level in enumerate(LEVEL_ORDER):
            for c, dim in enumerate(DIMENSIONS):
                p = cell_paths.get((level, dim))
                cell_rows.append({
                    "panel_type": "generated_sweep_3x6",
                    "experiment": experiment,
                    "ckpt_tag": ckpt_tag,
                    "image_id": image_id,
                    "level": level,
                    "dimension": dim,
                    "row_index": r,
                    "col_index": c,
                    "row_label": LEVEL_LABELS.get(level, level.title()),
                    "col_label": DIM_LABELS.get(dim, dim.title()),
                    "source_path": str(p) if p is not None else "",
                    "source_exists": bool(p is not None and Path(p).exists()),
                    "selection_note": cell_notes.get((level, dim), ""),
                    "panel_path": str(out_path),
                })

    return summary_rows, cell_rows


def main() -> None:
    ensure_dir(PANEL_ROOT)

    sweep_dirs = find_sweep_dirs(OUTPUT_ROOT)
    print(f"[INFO] Found {len(sweep_dirs)} sweep checkpoint folders.")

    all_summary = []
    all_cells = []

    for experiment, ckpt_tag, sweep_dir in sweep_dirs:
        try:
            summary_rows, cell_rows = build_panels_for_sweep_dir(
                experiment=experiment,
                ckpt_tag=ckpt_tag,
                sweep_dir=sweep_dir,
            )
            all_summary.extend(summary_rows)
            all_cells.extend(cell_rows)
        except Exception as e:
            print(f"[ERROR] Failed at {experiment}/{ckpt_tag}: {e}")

    summary_path = PANEL_ROOT / "generated_sweep_panel_summary.csv"
    cell_index_path = PANEL_ROOT / "generated_sweep_panel_cell_index.csv"

    pd.DataFrame(all_summary).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(all_cells).to_csv(cell_index_path, index=False, encoding="utf-8-sig")

    print("\n[DONE]")
    print(f"Generated panels: {len(all_summary)}")
    print(f"Panel root      : {PANEL_ROOT}")
    print(f"Summary CSV     : {summary_path}")
    print(f"Cell index CSV  : {cell_index_path}")


if __name__ == "__main__":
    main()
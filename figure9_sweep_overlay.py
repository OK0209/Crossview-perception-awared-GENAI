#!/usr/bin/env python3
"""已有 sweep 输出的 middle-difference 叠加图。只修改顶部 USER SETTINGS。
运行：python figure9_sweep_overlay.py。此文件也提供 reconstruction 绘图函数。
每个 ID/感知维度一张图；不运行 sweep 推理，不读取重建图作为 sweep 背景。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from PIL import Image

plt.rcParams["svg.fonttype"] = "none"

# ================= USER SETTINGS：只改这里 =================
BASE_DATA_DIR = Path("/home/yuebing/GenAI_density_urbanization/urban_data")
# 本次参考代码 OUTPUT_ROOT 含 /pre，因此实际 sweep 子目录也含 /pre。
# 若已有结果在 inference_outputs_multidim_ablation/sweep，直接修改此行。
INPUT_ROOT = BASE_DATA_DIR / "inference_outputs_multidim_ablation/sweep"
OUTPUT_ROOT = BASE_DATA_DIR / "figure9_sweep_middle"
FIGURE_OUTPUT_DIR = OUTPUT_ROOT / "figures"
IMAGE_IDS = [
    "514133a2fdc9f04926004394"
]
# 显示名 -> 推理目录中的实验名；每个实验选择一个 checkpoint。
# 可用实验：baseline / view_only / view_level_only / full。
MODELS = {"Score": "baseline", "View-Score": "view_only"}
CHECKPOINT_TAGS = {"baseline": None, "view_only": None}
# None 仅在选中样本只有一个 checkpoint 时自动选择。
# 有多个时填写 ckpt_tag，例如 "epoch19-step135840"（不带 .ckpt）。
TARGET_DIMS = ["beautiful"]  # 可改为 ["beautiful", "safety", "lively", "wealthy"]
LEVELS = ["low", "medium", "high"]
LEFT_GENERATION_LEVEL = "high"
STAGES = [(0.3,"Early"), (0.6,"Middle"), (0.9,"Late")]
OVERLAY_ALPHA = 0.45
COLORMAP = "jet"
NORMALIZATION = "per_map"  # per_map 或 shared
MAP_SIZE = 512
DPI = 300
TIMESTEP_TOLERANCE = 0.01
REFERENCE_PATHS = {"ground_truth": None, "bev": None, "view": None}
PATH_REMAP = []  # 目录搬迁时可填 [("/旧路径", "/新路径")]
# ==========================================================

MAP_TYPE = "control_delta_middle"
GENERATED_COLUMNS = ["generated_path", "generated_image_path", "output_image_path"]
REFERENCE_COLUMNS = {
    "ground_truth": ["real_path", "ground_truth_path"],
    "bev": ["bev_hint_path", "raw_hint_path", "hint_path"],
    "view": ["view_hint_path", "projected_hint_path"],
}


def text(value):
    return "" if value is None or pd.isna(value) else str(value).strip()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._") or "figure"


def resolve_path(value, source, root):
    """Resolve recorded paths; never select a random file by basename."""
    value = text(value)
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    candidates = []
    for old, new in PATH_REMAP:
        old = str(old).rstrip("/")
        if value == old or value.startswith(old + "/"):
            candidates.append(Path(str(new) + value[len(old):]))
    if not path.is_absolute():
        candidates += [Path(source).parent / path, root / path]
    # Canonical outputs can be relocated while retaining their /sweep/ suffix.
    for mode in ("sweep", "reconstruction"):
        if root.name == mode and f"/{mode}/" in value:
            candidates.append(root / value.split(f"/{mode}/", 1)[1])
    candidates.append(Path(source).parent / path.name)
    existing = sorted({p.resolve() for p in candidates if p.is_file()})
    if len(existing) > 1:
        raise RuntimeError(f"Ambiguous relocated path {value}: {existing}")
    return existing[0] if existing else None


def image_path(row, columns, root):
    for column in columns:
        path = resolve_path(row.get(column), row["_source"], root)
        if path:
            return path
    raise FileNotFoundError(
        f"Image not found for {row.get('experiment')}/{row.get('target_level')}; "
        f"checked {columns}. Metadata: {row['_source']}. "
        "Check PATH_REMAP or REFERENCE_PATHS for moved/missing references."
    )


def checkpoint_tag(row, experiment):
    direct = text(row.get("ckpt_tag"))
    if direct:
        return direct
    ckpt = text(row.get("ckpt_path"))
    if ckpt:
        return Path(ckpt).stem
    parts = Path(row["_source"]).parts
    if experiment in parts:
        pos = parts.index(experiment)
        if pos + 1 < len(parts) - 1:
            return parts[pos + 1]
    raise ValueError(f"Cannot determine checkpoint: {row['_source']}")


def load_selected_metadata(root, image_id, dimension, mode="sweep"):
    if not root.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {root}")
    wanted = set(MODELS.values())
    model_dirs = [root / exp for exp in wanted]
    scan_dirs = model_dirs if all(p.is_dir() for p in model_dirs) else [root]
    csvs = sorted({p for folder in scan_dirs
                   for p in folder.rglob("control_analysis_metadata.csv")})
    if not csvs:
        raise FileNotFoundError(f"No control_analysis_metadata.csv under {root}. "
            "For reconstruction, first run figure9_reconstruction_inference.py; "
            "generated reconstruction images alone are insufficient.")
    frames = []
    for i, csv in enumerate(csvs, 1):
        try:
            chunks = pd.read_csv(csv, dtype=str, keep_default_na=False,
                                 chunksize=50000)
            for chunk in chunks:
                needed = {"image_id", "analysis_type",
                          "timestep_fraction", "control_map_npy_path"}
                if mode == "sweep":
                    needed |= {"target_dim", "target_level"}
                if not needed.issubset(chunk.columns):
                    raise ValueError(f"Missing columns {needed - set(chunk.columns)}")
                mask = (chunk["image_id"] == image_id) & (chunk["analysis_type"] == MAP_TYPE)
                if mode == "sweep":
                    mask &= chunk["target_dim"] == dimension
                if "mode" in chunk:
                    mask &= chunk["mode"].isin(["", mode])
                elif mode == "reconstruction" and "reconstruction" not in csv.parts:
                    raise ValueError("Cannot verify reconstruction source: missing mode field "
                                     "and no reconstruction directory in the metadata path.")
                chunk = chunk[mask].copy()
                if mode == "reconstruction":
                    # No perceptual-level sweep is carried out for reconstruction.
                    chunk["target_level"] = "original"
                    if "target_dim" not in chunk:
                        chunk["target_dim"] = "reconstruction"
                if chunk.empty:
                    continue
                if "experiment" not in chunk:
                    matches = wanted.intersection(csv.parts)
                    if len(matches) != 1:
                        raise ValueError("Cannot infer experiment from metadata path")
                    chunk["experiment"] = next(iter(matches))
                chunk = chunk[chunk["experiment"].isin(wanted)].copy()
                if not chunk.empty:
                    chunk["_source"] = str(csv)
                    frames.append(chunk)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            raise ValueError(f"Cannot read metadata {csv}: {exc}") from exc
        if i % 200 == 0:
            print(f"[SCAN] {i}/{len(csvs)} metadata files", flush=True)
    if not frames:
        raise RuntimeError(
            f"No {MAP_TYPE} rows for image_id={image_id}, target_dim={dimension}. "
            "Check the ID, dimension and selected sweep folder. This script does "
            "not substitute control_residual maps or re-run inference."
        )
    df = pd.concat(frames, ignore_index=True)
    df["target_level"] = df["target_level"].str.lower().str.strip()
    df["timestep_fraction"] = pd.to_numeric(df["timestep_fraction"], errors="raise")
    df["_checkpoint"] = [checkpoint_tag(row, row["experiment"])
                          for row in df.to_dict("records")]
    selected = []
    for exp in MODELS.values():
        sub = df[df["experiment"] == exp].copy()
        tags = sorted(sub["_checkpoint"].unique().tolist())
        chosen = CHECKPOINT_TAGS.get(exp)
        if chosen is None:
            if len(tags) != 1:
                raise RuntimeError(f"{exp}: checkpoint candidates={tags}. "
                                   "Set CHECKPOINT_TAGS at the top of this file.")
            chosen = tags[0]
        sub = sub[sub["_checkpoint"] == chosen]
        if sub.empty:
            raise RuntimeError(f"{exp}: checkpoint {chosen!r} not found; available={tags}")
        selected.append(sub)
    return pd.concat(selected, ignore_index=True)


def select_record(df, experiment, level, fraction, root):
    rows = df[(df["experiment"] == experiment) & (df["target_level"] == level)
              & ((df["timestep_fraction"] - fraction).abs() <= TIMESTEP_TOLERANCE)]
    if rows.empty:
        raise RuntimeError(f"Missing middle-difference map: {experiment}/{level}/{fraction}")
    candidates = {}
    for row in rows.to_dict("records"):
        path = resolve_path(row["control_map_npy_path"], row["_source"], root)
        if path is None:
            raise FileNotFoundError(f"Missing .npy: {row['control_map_npy_path']}")
        if path.suffix.lower() != ".npy":
            raise ValueError(f"Expected numeric .npy map, not a colored heatmap: {path}")
        generated = image_path(row, GENERATED_COLUMNS, root)
        key = (str(path), str(generated), float(row["timestep_fraction"]))
        row["_map_path"] = str(path)
        row["_generated_path"] = str(generated)
        candidates[key] = row  # Collapse combined/per-sample duplicate metadata.
    if len(candidates) != 1:
        raise RuntimeError(f"Ambiguous map/image pair: {experiment}/{level}/{fraction}: "
                           f"{list(candidates)}. Select one checkpoint/sample first.")
    return next(iter(candidates.values()))


def load_map(path):
    arr = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if arr.ndim != 2 or not np.isfinite(arr).all() or arr.min() < 0:
        raise ValueError(f"Expected finite, nonnegative 2D saved channel-RMS map: {path}")
    # Resize before coloring, matching the existing bilinear visualization.
    return np.asarray(Image.fromarray(arr).resize((MAP_SIZE, MAP_SIZE),
                      resample=Image.Resampling.BILINEAR), dtype=np.float32)


def load_image(path):
    with Image.open(path) as im:
        if im.mode == "RGBA" or "transparency" in im.info:
            im = im.convert("RGBA")
            background = Image.new("RGBA", im.size, "white")
            background.alpha_composite(im)
            im = background.convert("RGB")
        else:
            im = im.convert("RGB")
        return np.asarray(im.resize((MAP_SIZE, MAP_SIZE), Image.Resampling.BICUBIC))


def overlay_image(background, arr, bounds):
    lo, hi = bounds
    relative = np.zeros_like(arr) if hi <= lo else np.clip((arr-lo)/(hi-lo), 0, 1)
    colors = plt.get_cmap(COLORMAP)(relative)[..., :3]
    mixed = (1-OVERLAY_ALPHA) * background.astype(float)/255 + OVERLAY_ALPHA * colors
    return np.clip(np.rint(mixed * 255), 0, 255).astype(np.uint8)


def render(root, output, image_id, dimension, levels, left_level, mode="sweep"):
    if not 0 <= OVERLAY_ALPHA <= 1:
        raise ValueError("OVERLAY_ALPHA must be between 0 and 1")
    if NORMALIZATION not in {"per_map", "shared"}:
        raise ValueError("NORMALIZATION must be per_map or shared")
    fractions = [float(fraction) for fraction, _ in STAGES]
    if len(fractions) != 3 or fractions != sorted(set(fractions)) or 0.6 not in fractions:
        raise ValueError("STAGES must contain three increasing fractions including 0.6")
    if mode == "reconstruction":
        levels, left_level, dimension = ["original"], "original", "reconstruction"
    df = load_selected_metadata(root, image_id, dimension, mode)
    records, arrays, image_cache = {}, {}, {}
    for label, experiment in MODELS.items():
        for level in dict.fromkeys([*levels, left_level]):
            for fraction, _ in STAGES:
                # References need only the middle record for the extra left level.
                if level not in levels and fraction != 0.6:
                    continue
                record = select_record(df, experiment, level, fraction, root)
                key = (label, level, fraction)
                records[key] = record
                if level in levels:
                    arrays[key] = load_map(record["_map_path"])
        for level in levels:
            backgrounds = {records[(label, level, f)]["_generated_path"] for f, _ in STAGES}
            if len(backgrounds) != 1:
                raise RuntimeError(f"Stages do not refer to the same final image: {label}/{level}")
    # In the shared-scale mode, one numeric range is used across all panels.
    shared_bounds = (min(float(a.min()) for a in arrays.values()),
                     max(float(a.max()) for a in arrays.values()))
    left_records = {label: records[(label, left_level, 0.6)] for label in MODELS}
    score = next(iter(left_records.values()))
    view = next((row for row in left_records.values()
                 if text(row.get("view_hint_path")) or text(row.get("projected_hint_path"))), score)
    refs = {}
    for name in REFERENCE_COLUMNS:
        explicit = REFERENCE_PATHS[name]
        if explicit:
            path = Path(explicit).expanduser()
            if not path.is_file():
                raise FileNotFoundError(path)
            refs[name] = path.resolve()
        else:
            row = view if name == "view" else score
            refs[name] = image_path(row, REFERENCE_COLUMNS[name], root)
    # Check the two experiments use the same structural input when recorded.
    score_bev = image_path(score, REFERENCE_COLUMNS["bev"], root)
    for label, row in left_records.items():
        other_bev = image_path(row, REFERENCE_COLUMNS["bev"], root)
        if score_bev != other_bev and not np.array_equal(load_image(score_bev), load_image(other_bev)):
            raise RuntimeError(f"Selected models have different BEV inputs ({label}); do not pair them.")

    is_reconstruction = mode == "reconstruction"
    # Reconstruction has one condition per model: put models on separate rows,
    # keeping three stage columns. Sweep retains model groups side by side.
    fig = plt.figure(figsize=(9 if is_reconstruction else 3+6*len(MODELS),
                     max(7.4, len(MODELS)*3 if is_reconstruction else len(levels)*2.7)), facecolor="white")
    outer = fig.add_gridspec(1, 2 if is_reconstruction else 1+len(MODELS),
                            width_ratios=[1.03, 3] if is_reconstruction else [1.03] + [3]*len(MODELS),
                            left=.035, right=.985, bottom=.17, top=.88, wspace=.23)
    model_rows = outer[1].subgridspec(len(MODELS), 1, hspace=.4) if is_reconstruction else None
    left_grid = outer[0].subgridspec(3+len(MODELS), 1, hspace=.32)
    ref_items = [(refs["ground_truth"], "Ground truth"), (refs["bev"], "BEV"),
                 (refs["view"], "Projected view hint")]
    ref_items.extend((Path(row["_generated_path"]), f"{label} reconstruction" if mode == "reconstruction"
                      else f"{label} generation ({left_level})") for label,row in left_records.items())
    for i, (path, title) in enumerate(ref_items):
        ax = fig.add_subplot(left_grid[i])
        ax.imshow(load_image(path)); ax.set_axis_off()
        ax.text(.5, -.045, title, transform=ax.transAxes, ha="center", va="top", fontsize=9)

    manifest = []
    group_axes = []
    for g, (label, experiment) in enumerate(MODELS.items(), 1):
        model_cell = model_rows[g-1] if is_reconstruction else outer[g]
        grid = model_cell.subgridspec(len(levels), 3, hspace=.12, wspace=.018)
        axes = []
        for r, level in enumerate(levels):
            for c, (fraction, stage) in enumerate(STAGES):
                key = (label, level, fraction)
                record, arr = records[key], arrays[key]
                gp = record["_generated_path"]
                if gp not in image_cache:
                    image_cache[gp] = load_image(gp)
                bounds = shared_bounds if NORMALIZATION == "shared" else (float(arr.min()), float(arr.max()))
                ax = fig.add_subplot(grid[r, c])
                ax.imshow(overlay_image(image_cache[gp], arr, bounds)); ax.set_axis_off()
                if r == 0:
                    ax.set_title(f"{stage}\n({fraction:.1f})", fontsize=11, pad=8)
                if c == 0:
                    ax.text(-.055, .5, level.capitalize(), transform=ax.transAxes,
                            rotation=90, va="center", ha="right", fontsize=11)
                axes.append(ax)
                manifest.append({"model_label": label, "experiment": experiment,
                    "mode": mode,
                    "ckpt_tag": record["_checkpoint"], "image_id": image_id,
                    "target_dim": dimension, "target_level": level,
                    "analysis_type": MAP_TYPE, "timestep_fraction": fraction,
                    "map_npy_path": record["_map_path"], "generated_path": gp,
                    "metadata_source": record["_source"],
                    "normalization": NORMALIZATION, "display_min": bounds[0],
                    "display_max": bounds[1], "overlay_alpha": OVERLAY_ALPHA,
                    "ddim_step_index": record.get("ddim_step_index", ""),
                    "original_timestep_value": record.get("original_timestep_value", "")})
        group_axes.append((label, axes))
    fig.canvas.draw()
    for label, axes in group_axes:
        x0 = min(a.get_position().x0 for a in axes)
        x1 = max(a.get_position().x1 for a in axes)
        y0 = min(a.get_position().y0 for a in axes)
        fig.text((x0+x1)/2, y0-.026, label, ha="center", va="top", fontsize=14, fontweight="bold")
    first_axes = group_axes[0][1]; last_axes = group_axes[-1][1]
    x0 = min(a.get_position().x0 for a in first_axes)
    x1 = max(a.get_position().x1 for a in last_axes)
    cax = fig.add_axes([x0, .087, x1-x0, .014])
    sm = ScalarMappable(norm=Normalize(0,1), cmap=COLORMAP)
    bar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0,.5,1])
    bar.ax.tick_params(labelsize=9)
    bar.set_label("Relative middle-feature difference (per-map min-max)" if NORMALIZATION == "per_map"
                  else "Middle-feature difference (shared min-max scale)", fontsize=10)
    context = "Reconstruction (original conditions)" if mode == "reconstruction" else dimension
    fig.suptitle(f"Middle-feature-difference overlays | {context} | {image_id}", fontsize=14, y=.97)
    fig.text(.5,.017,"Maps are overlaid on each model's final reconstruction." if is_reconstruction else
             "All stages are overlaid on the corresponding final generated image; "
             "left generation thumbnails show " + left_level + ".",ha="center",fontsize=9,color="#555555")
    output.mkdir(parents=True, exist_ok=True)
    stem = f"figure9_{mode}_{safe_name(image_id)}_{safe_name(dimension)}_{'-'.join(levels)}"
    for suffix in ("png", "pdf", "svg"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
        print(f"[SAVE] {path}")
    plt.close(fig)
    pd.DataFrame(manifest).to_csv(output / f"{stem}_sources.csv", index=False)
    settings = {"input_root": str(root), "image_id": image_id, "mode": mode, "target_dim": dimension,
        "levels": levels, "left_generation_level": left_level, "stages": STAGES,
        "models": MODELS, "normalization": NORMALIZATION, "colormap": COLORMAP,
        "overlay_alpha": OVERLAY_ALPHA, "map_size": MAP_SIZE,
        "reference_paths": {k:str(v) for k,v in refs.items()},
        "description": "Saved channel-RMS middle delta, overlaid on final generated images."}
    (output / f"{stem}_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return output / f"{stem}.png"



def main():
    """直接使用顶部配置，批量绘制已有 sweep maps。"""
    if not IMAGE_IDS or not TARGET_DIMS or not LEVELS or not MODELS:
        raise ValueError("IMAGE_IDS / TARGET_DIMS / LEVELS / MODELS 不能为空")
    if len(set(MODELS.values())) != len(MODELS):
        raise ValueError("每个实验只能配置一个模型；不同 checkpoint 请分别出图")
    status=[]
    for image_id in dict.fromkeys(IMAGE_IDS):
        for dimension in dict.fromkeys(TARGET_DIMS):
            try:
                path=render(INPUT_ROOT.expanduser().resolve(), FIGURE_OUTPUT_DIR.expanduser().resolve(),
                            image_id,dimension,LEVELS,LEFT_GENERATION_LEVEL,"sweep")
                status.append({"image_id":image_id,"target_dim":dimension,"status":"ok","figure":str(path),"error":""})
            except (ValueError,RuntimeError,FileNotFoundError) as exc:
                print(f"[FAILED] {image_id}/{dimension}: {exc}",flush=True)
                status.append({"image_id":image_id,"target_dim":dimension,"status":"failed","figure":"","error":str(exc)})
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(status)
    table.to_csv(OUTPUT_ROOT/"batch_status.csv",index=False)
    if (table.status=="failed").any():
        raise SystemExit("部分出图失败，查看 OUTPUT_ROOT/batch_status.csv")
    print(f"[DONE] sweep figures: {FIGURE_OUTPUT_DIR}")


if __name__=="__main__":
    main()

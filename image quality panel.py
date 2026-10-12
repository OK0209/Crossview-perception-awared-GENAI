#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a one-image Score vs View-Score reconstruction analysis and build one paper panel.

This script reuses the validated model loading, prompt construction, sampling,
view projection, and ControlNet feature capture in
``perception_inference_multi_dim_cross*.py``.  Edit USER SETTINGS below and
run this file directly; no command-line arguments are required.

Model-name mapping used by the current ablation code:
    Score      -> experiment ``baseline``  (raw score prompt, no view transform)
    View-Score -> experiment ``view_only`` (raw score prompt + view transform)

The final panel contains, from left to right:
    GT / BEV / projected View / Score generation / View-Score generation
    Score residual-response overlays: Early / Medium / Late
    View-Score residual-response overlays: Early / Medium / Late

Only ControlNet residual-response maps are shown.  A cell is the channel-wise
RMS response of one scaled ControlNet residual tensor, min-max normalized for
visualization and overlaid on that model's generated image by the inference
script.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


# ============================================================
# 0. USER SETTINGS：只修改这里，然后直接运行本文件
# ============================================================

# "all"   : 先运行两个模型，再生成组合图（通常使用这个）
# "infer" : 只运行两个模型并保存 feature maps
# "panel" : 不运行模型，只使用已经存在的结果重新拼图
RUN_MODE = "all"

# 原始推理代码的位置。文件名中有空格或括号也没有问题。
INFERENCE_SCRIPT = Path(
    "/home/yuebing/GenAI_density_urbanization/"
    "perception_inference_multi_dim_cross copy.py"
)

# 可以填写纯 image id，也可以填写带扩展名的图片名。
TARGET_IMAGE_NAME = "51421849fdc9f0492600810d"

# Reconstruction 使用该图片在 CSV 中原有的全部六维感知值，不进行
# low / medium / high sweep。这里的顺序应与原推理代码保持一致。
RECONSTRUCTION_DIMENSIONS = [
    "beautiful", "safety", "wealthy", "boring", "depressing", "lively"
]

# Score 对应 baseline；View-Score 对应 view_only。
# 可填写 "best"、"last.ckpt"、"epoch_latest" 或具体 checkpoint 文件名。
SCORE_CKPT = "epoch19-step135840.ckpt"
VIEW_SCORE_CKPT = "epoch19-step135840.ckpt"

OUTPUT_ROOT = Path(
    "/home/yuebing/GenAI_density_urbanization/urban_data/"
    "inference_outputs_multidim_ablation/recon_feature_pannel"
)

# panel 模式下，如果同一模型存在多个 checkpoint 输出：
# - None：自动选择最近完成的输出；
# - 字符串：指定 checkpoint 输出文件夹名，例如 "epoch19-step135840"。
SCORE_OUTPUT_TAG: Optional[str] = None
VIEW_SCORE_OUTPUT_TAG: Optional[str] = None

# 最终论文图设置。
BLOCKS = [0, 3, 6, 9, 12]
STAGE_FRACTIONS = [0.3, 0.6, 0.9]  # Early / Medium / Late
RENDER_MODE = "overlay"             # "overlay" or "heatmap"
CELL_SIZE = 256
OUTPUT_PATH: Optional[Path] = None   # None 表示自动命名和保存

DEVICE = "auto"  # "auto", "cuda:0", or "cpu"

# 建议保持 True。若生成图已经存在但 feature maps 不完整，False 会跳过重采样。
OVERWRITE = True

# 下面路径保持 None 时，直接沿用原始推理代码中的设置。
STANDARDIZED_CSV: Optional[Path] = None
IMAGE_DIR: Optional[Path] = None
LAYOUT_DIR: Optional[Path] = None
MODEL_CONFIG_PATH: Optional[Path] = None
CKPT_ROOT: Optional[Path] = None

# ============================================================
# USER SETTINGS END
# ============================================================

SCORE_EXPERIMENT = "baseline"
VIEW_SCORE_EXPERIMENT = "view_only"
STAGE_LABELS = ("Early", "Medium", "Late")


def image_id_from_name(value: str) -> str:
    """Accept either an image id or a filename such as ``abc123.jpg``."""
    return Path(str(value).strip()).stem


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_inference_module(script_path: Path):
    script_path = script_path.expanduser().resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Inference script does not exist: {script_path}")

    module_name = "perception_inference_cross_runtime"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import inference script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def apply_optional_path(module, attr: str, value: Optional[Path]) -> None:
    if value is not None:
        setattr(module, attr, Path(value).expanduser())


def sample_with_control_only(inference, **sample_kwargs):
    """
    Call the existing sampler without entering its cross-attention token selector.

    Some inference-script versions calculate ``select_token_indices`` even when
    ``capture_attention=False``. That path is unnecessary for a ControlNet-only
    reconstruction and older copies can fail there because ``cleaned`` is not
    defined. The selected-token result is never used when no attention recorder
    is created, so returning an empty selection is behaviorally equivalent.
    """
    original_selector = inference.select_token_indices

    def empty_token_selection(*_args, **_kwargs):
        return {}

    inference.select_token_indices = empty_token_selection
    try:
        return inference.sample_images_with_cross_attention(**sample_kwargs)
    finally:
        inference.select_token_indices = original_selector


def run_reconstruction_analysis_for_experiment(
    inference,
    cfg,
    df: pd.DataFrame,
    experiment: str,
    ckpt_selector: str,
    device,
) -> Tuple[pd.DataFrame, str]:
    """Reconstruct the selected image and capture ControlNet residual responses."""
    model, sampler, ckpt_path = inference.load_model_and_sampler(
        cfg, experiment, ckpt_selector, device
    )
    ckpt_tag = ckpt_path.stem
    use_view_transform = bool(inference.EXPERIMENTS[experiment]["use_view_transform"])

    reconstruction_root = cfg.output_root / "reconstruction"
    run_dir = inference.ensure_dir(
        reconstruction_root / "analysis_runs" / experiment / ckpt_tag
    )
    generated_dir = inference.ensure_dir(run_dir / "generated")
    prompt_dir = inference.ensure_dir(run_dir / "prompts")
    control_root = inference.ensure_dir(run_dir / "control_analysis")
    real_dir = inference.ensure_dir(reconstruction_root / "real")
    bev_dir = inference.ensure_dir(reconstruction_root / "BEV")

    selected_rows = inference.select_reconstruction_rows(df, cfg)
    metadata_rows: List[Dict] = []
    all_control_rows: List[Dict] = []

    for _, row in selected_rows.iterrows():
        image_name = str(row["_image_name_raw"])
        hint_name = str(row["_hint_name_raw"])
        image_id = str(row["_image_id"])

        image_path = (
            inference.find_file(cfg.image_dir, image_name)
            or inference.find_file(cfg.image_dir, image_id)
        )
        hint_path = (
            inference.find_file(cfg.layout_dir, hint_name)
            or inference.find_file(cfg.layout_dir, inference.stem_no_ext(hint_name))
            or inference.find_file(cfg.layout_dir, image_id)
        )
        if image_path is None or hint_path is None:
            print(f"[RECON-SKIP] missing image or BEV: {experiment}/{image_id}")
            continue

        real_path = real_dir / f"{image_id}.png"
        bev_path = bev_dir / f"{image_id}.png"
        inference.safe_copy_or_save_image(image_path, real_path, cfg.image_resolution)
        inference.safe_copy_or_save_image(hint_path, bev_path, cfg.image_resolution)

        try:
            control = inference.process_hint_image(hint_path, cfg, device)
        except Exception as exc:
            print(f"[RECON-SKIP] {experiment}/{image_id}: {exc}")
            continue

        view_hint_path = ""
        view_hint_error = ""
        if inference.SAVE_VIEW_HINT:
            view_hint_path, view_hint_error = inference.get_or_save_shared_view_hint(
                model=model,
                control=control,
                out_base=reconstruction_root,
                image_id=image_id,
                use_view_transform=use_view_transform,
                image_resolution=cfg.image_resolution,
                overwrite=cfg.overwrite,
            )
            if view_hint_error:
                print(f"[VIEW-HINT-WARN] {experiment}/{image_id}: {view_hint_error}")

        # No override is supplied: this is the original reconstruction prompt
        # built from all six scores stored in this image's CSV row.
        prompt = inference.build_prompt_for_experiment(row, cfg, experiment)
        prompt_path = prompt_dir / f"{image_id}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        generated_path = generated_dir / f"{image_id}.png"
        control_dir = inference.ensure_dir(control_root / image_id)
        per_image_control_meta = control_dir / "control_analysis_metadata.csv"
        need_sampling = (
            cfg.overwrite
            or not generated_path.exists()
            or not per_image_control_meta.exists()
        )
        control_rows: List[Dict] = []
        control_metadata_path = str(per_image_control_meta)
        control_errors: List[str] = []

        if need_sampling:
            sample_seed = cfg.seed + inference.stable_int_hash(f"recon-{image_id}")
            inference.set_global_seed(sample_seed)

            # The existing recorder needs an image on disk before it saves the
            # overlays. Use a temporary current-generation image so OVERWRITE
            # never overlays a new heatmap on an older reconstruction.
            temporary_generation = run_dir / f".{image_id}_{experiment}_overlay_source.png"
            if temporary_generation.exists():
                temporary_generation.unlink()

            sample_meta = {
                "mode": "reconstruction",
                "experiment": experiment,
                "ckpt_tag": ckpt_tag,
                "image_id": image_id,
                "generated_path": str(generated_path),
                "real_path": str(real_path),
                "bev_hint_path": str(bev_path),
                "view_hint_path": view_hint_path,
                "raw_hint_path": str(hint_path),
                "prompt": prompt,
                # Kept blank because reconstruction does not sweep one target
                # dimension or one ordinal level.
                "target_dim": "",
                "target_level": "",
            }
            try:
                (
                    images,
                    _attention_rows,
                    _attention_metadata_path,
                    control_rows,
                    control_metadata_path,
                    control_errors,
                ) = sample_with_control_only(
                    inference=inference,
                    model=model,
                    sampler=sampler,
                    control=control,
                    prompt=prompt,
                    cfg=cfg,
                    attention_out_dir=None,
                    sample_meta=sample_meta,
                    generated_path_for_overlay=temporary_generation,
                    control_out_dir=control_dir,
                    raw_hint_path=hint_path,
                    view_hint_path=Path(view_hint_path) if view_hint_path else None,
                    capture_attention=False,
                    capture_control=True,
                )
                inference.save_generated_images(
                    images, generated_path, overwrite=cfg.overwrite
                )
            finally:
                if temporary_generation.exists():
                    temporary_generation.unlink()
        else:
            control_rows = pd.read_csv(per_image_control_meta).to_dict("records")

        all_control_rows.extend(control_rows)
        metadata = {
            "mode": "reconstruction",
            "experiment": experiment,
            "ckpt_selector": ckpt_selector,
            "ckpt_path": str(ckpt_path),
            "ckpt_tag": ckpt_tag,
            "image_id": image_id,
            "image_name_raw": image_name,
            "generated_path": str(generated_path),
            "real_path": str(real_path),
            "bev_hint_path": str(bev_path),
            "source_hint_path": str(hint_path),
            "view_hint_path": view_hint_path,
            "view_hint_error": view_hint_error,
            "use_view_transform": use_view_transform,
            "control_analysis_dir": str(control_dir),
            "control_analysis_metadata_path": control_metadata_path,
            "control_analysis_error": " | ".join(control_errors),
            "prompt_path": str(prompt_path),
            "prompt": prompt,
        }
        for dimension in cfg.dimensions:
            metadata[f"{dimension}_score"] = inference.get_dim_score(row, dimension)
            metadata[f"{dimension}_level"] = inference.get_dim_level(row, dimension)
        metadata_rows.append(metadata)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = run_dir / "metadata_reconstruction.csv"
    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    if all_control_rows:
        pd.DataFrame(all_control_rows).to_csv(
            run_dir / "control_analysis_metadata.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"[RECON] metadata saved: {metadata_path}")

    del model, sampler
    if inference.torch.cuda.is_available():
        inference.torch.cuda.empty_cache()
    return metadata_df, ckpt_tag


def run_one_image_inference() -> Dict[str, str]:
    """Run Score and View-Score reconstruction for one selected image."""
    inference = load_inference_module(INFERENCE_SCRIPT)
    image_id = image_id_from_name(TARGET_IMAGE_NAME)

    inference.RUN_MODE = "reconstruction"
    inference.EXPERIMENT_ORDER = [SCORE_EXPERIMENT, VIEW_SCORE_EXPERIMENT]
    inference.DIMENSIONS = list(RECONSTRUCTION_DIMENSIONS)
    inference.TARGET_IMAGE_IDS = [image_id]
    inference.OUTPUT_ROOT = Path(OUTPUT_ROOT).expanduser()
    inference.DEVICE = DEVICE
    inference.OVERWRITE = bool(OVERWRITE)
    inference.SAVE_VIEW_HINT = True
    inference.SAVE_CROSS_ATTENTION = False
    inference.SAVE_CONTROL_ANALYSIS = True
    inference.ANALYSIS_TIMESTEP_FRACTIONS = list(STAGE_FRACTIONS)
    inference.CONTROL_ANALYSIS_MAX_BLOCKS = max(BLOCKS) + 1
    inference.CONTROL_ANALYSIS_SAVE_NUMPY = True
    inference.CONTROL_ANALYSIS_SAVE_PNG = True
    inference.CONTROL_ANALYSIS_SAVE_OVERLAY = True
    inference.CONTROL_ANALYSIS_SAVE_OVERLAY_ON_VIEWHINT = True

    apply_optional_path(inference, "STANDARDIZED_CSV", STANDARDIZED_CSV)
    apply_optional_path(inference, "IMAGE_DIR", IMAGE_DIR)
    apply_optional_path(inference, "LAYOUT_DIR", LAYOUT_DIR)
    apply_optional_path(inference, "MODEL_CONFIG_PATH", MODEL_CONFIG_PATH)
    apply_optional_path(inference, "CKPT_ROOT", CKPT_ROOT)

    cfg = inference.build_config_from_user_settings()
    device = inference.get_device(cfg.device)
    inference.set_global_seed(cfg.seed)
    inference.ensure_dir(cfg.output_root)
    inference.print_config(cfg, device)
    df, image_col = inference.load_label_table(cfg)
    print(f"[DATA] image column: {image_col}")

    selectors = {
        SCORE_EXPERIMENT: SCORE_CKPT,
        VIEW_SCORE_EXPERIMENT: VIEW_SCORE_CKPT,
    }
    output_tags: Dict[str, str] = {}
    all_metadata: List[pd.DataFrame] = []
    for experiment in (SCORE_EXPERIMENT, VIEW_SCORE_EXPERIMENT):
        metadata, ckpt_tag = run_reconstruction_analysis_for_experiment(
            inference=inference,
            cfg=cfg,
            df=df,
            experiment=experiment,
            ckpt_selector=selectors[experiment],
            device=device,
        )
        output_tags[experiment] = ckpt_tag
        all_metadata.append(metadata)

    if all_metadata:
        combined = pd.concat(all_metadata, ignore_index=True)
        combined_path = (
            cfg.output_root
            / "reconstruction"
            / "metadata_score_vs_view_score_reconstruction.csv"
        )
        combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"[DONE] combined metadata saved: {combined_path}")
    return output_tags


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=3)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = x0 + (x1 - x0 - width) / 2 - bbox[0]
    y = y0 + (y1 - y0 - height) / 2 - bbox[1]
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center", spacing=3)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required metadata file does not exist: {path}")
    return pd.read_csv(path)


def find_run_dir(
    output_root: Path,
    experiment: str,
    requested_tag: Optional[str],
) -> Path:
    experiment_dir = output_root / "reconstruction" / "analysis_runs" / experiment
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment output directory does not exist: {experiment_dir}")

    if requested_tag:
        exact = experiment_dir / requested_tag
        if not exact.exists():
            raise FileNotFoundError(f"Checkpoint output directory does not exist: {exact}")
        return exact

    candidates = [
        path
        for path in experiment_dir.iterdir()
        if path.is_dir()
        and (path / "metadata_reconstruction.csv").exists()
        and (path / "control_analysis_metadata.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No completed control-analysis output found under {experiment_dir}"
        )
    chosen = max(candidates, key=lambda path: path.stat().st_mtime)
    print(f"[AUTO] {experiment} output tag: {chosen.name}")
    return chosen


def filter_sample_rows(
    df: pd.DataFrame,
    image_id: str,
) -> pd.DataFrame:
    required = {"image_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    return df[
        (df["image_id"].astype(str).map(image_id_from_name) == image_id)
    ].copy()


def first_existing_path(
    values: Iterable[object],
    run_dir: Path,
    local_search_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve absolute saved paths and tolerate moving an output tree."""
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        raw = Path(text).expanduser()
        direct_candidates = [raw]
        if not raw.is_absolute():
            direct_candidates.extend([run_dir / raw, Path.cwd() / raw])
        for candidate in direct_candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        # Metadata often stores an old absolute root.  The basename remains
        # stable, so try the known local analysis folder before giving up.
        if local_search_dir is not None and local_search_dir.exists():
            candidate = local_search_dir / raw.name
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def sample_path(
    sample_rows: pd.DataFrame,
    column: str,
    run_dir: Path,
    fallback: Optional[Path] = None,
) -> Optional[Path]:
    values = sample_rows[column].tolist() if column in sample_rows.columns else []
    found = first_existing_path(values, run_dir)
    if found is not None:
        return found
    if fallback is not None and fallback.exists():
        return fallback
    return None


def select_control_path(
    control_rows: pd.DataFrame,
    run_dir: Path,
    image_id: str,
    block: int,
    stage_fraction: float,
    render_mode: str,
) -> Tuple[Optional[Path], Optional[float]]:
    required = {
        "image_id",
        "analysis_type",
        "block_index",
        "timestep_fraction",
    }
    missing = required - set(control_rows.columns)
    if missing:
        raise ValueError(f"Control metadata is missing columns: {sorted(missing)}")

    rows = control_rows[
        (control_rows["image_id"].astype(str).map(image_id_from_name) == image_id)
        & (control_rows["analysis_type"].astype(str) == "control_residual")
        & (pd.to_numeric(control_rows["block_index"], errors="coerce") == int(block))
    ].copy()
    if rows.empty:
        return None, None

    fractions = pd.to_numeric(rows["timestep_fraction"], errors="coerce")
    valid = fractions.notna()
    rows = rows.loc[valid].copy()
    fractions = fractions.loc[valid]
    if rows.empty:
        return None, None

    nearest_index = (fractions - float(stage_fraction)).abs().idxmin()
    nearest_fraction = float(fractions.loc[nearest_index])
    nearest_rows = rows[
        np.isclose(
            pd.to_numeric(rows["timestep_fraction"], errors="coerce"),
            nearest_fraction,
        )
    ]

    analysis_dir = run_dir / "control_analysis" / image_id
    if render_mode == "heatmap":
        columns = ["control_map_png_path"]
    else:
        columns = ["control_overlay_path", "control_map_png_path"]

    for column in columns:
        if column not in nearest_rows.columns:
            continue
        path = first_existing_path(
            nearest_rows[column].tolist(),
            run_dir=run_dir,
            local_search_dir=analysis_dir,
        )
        if path is not None:
            return path, nearest_fraction
    return None, nearest_fraction


def fit_image(path: Optional[Path], size: Tuple[int, int]) -> Image.Image:
    if path is None or not path.exists():
        image = Image.new("RGB", size, (242, 242, 242))
        draw = ImageDraw.Draw(image)
        draw_centered(draw, (0, 0, size[0], size[1]), "Missing", get_font(22), (130, 130, 130))
        return image
    try:
        with Image.open(path) as source:
            return ImageOps.fit(
                source.convert("RGB"),
                size,
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
    except Exception:
        return fit_image(None, size)


def build_combined_panel(
    output_root: Path,
    image_id: str,
    blocks: Sequence[int],
    stage_fractions: Sequence[float],
    score_tag: Optional[str],
    view_score_tag: Optional[str],
    render_mode: str,
    cell_size: int,
    output_path: Optional[Path],
) -> Tuple[Path, Path]:
    image_id = image_id_from_name(image_id)
    output_root = output_root.expanduser()

    score_dir = find_run_dir(output_root, SCORE_EXPERIMENT, score_tag)
    view_score_dir = find_run_dir(output_root, VIEW_SCORE_EXPERIMENT, view_score_tag)

    score_meta = filter_sample_rows(
        read_csv(score_dir / "metadata_reconstruction.csv"),
        image_id,
    )
    view_score_meta = filter_sample_rows(
        read_csv(view_score_dir / "metadata_reconstruction.csv"),
        image_id,
    )
    if score_meta.empty:
        raise ValueError(f"No Score reconstruction found for {image_id} in {score_dir}")
    if view_score_meta.empty:
        raise ValueError(
            f"No View-Score reconstruction found for {image_id} in {view_score_dir}"
        )

    score_control = read_csv(score_dir / "control_analysis_metadata.csv")
    view_score_control = read_csv(view_score_dir / "control_analysis_metadata.csv")

    reconstruction_root = output_root / "reconstruction"
    references: List[Tuple[str, Optional[Path]]] = [
        (
            "Ground Truth",
            sample_path(
                score_meta,
                "real_path",
                score_dir,
                reconstruction_root / "real" / f"{image_id}.png",
            ),
        ),
        (
            "BEV",
            sample_path(
                score_meta,
                "bev_hint_path",
                score_dir,
                reconstruction_root / "BEV" / f"{image_id}.png",
            ),
        ),
        (
            "View",
            sample_path(
                view_score_meta,
                "view_hint_path",
                view_score_dir,
                reconstruction_root / "view_hint" / f"{image_id}_view_hint.png",
            ),
        ),
        (
            "Score\nGeneration",
            sample_path(score_meta, "generated_path", score_dir),
        ),
        (
            "View-Score\nGeneration",
            sample_path(view_score_meta, "generated_path", view_score_dir),
        ),
    ]

    if len(blocks) != len(references):
        raise ValueError(
            "The requested publication layout uses five reference rows, so "
            "BLOCKS must contain exactly five block indices."
        )
    if len(stage_fractions) != 3:
        raise ValueError("STAGE_FRACTIONS requires exactly three values: Early Medium Late")

    heatmap_paths: Dict[Tuple[str, int, int], Optional[Path]] = {}
    cell_index: List[Dict[str, object]] = []

    model_specs = [
        ("Score", score_dir, score_control),
        ("View-Score", view_score_dir, view_score_control),
    ]
    for model_label, run_dir, control_df in model_specs:
        for row_index, block in enumerate(blocks):
            for stage_index, requested_fraction in enumerate(stage_fractions):
                path, actual_fraction = select_control_path(
                    control_rows=control_df,
                    run_dir=run_dir,
                    image_id=image_id,
                    block=int(block),
                    stage_fraction=float(requested_fraction),
                    render_mode=render_mode,
                )
                heatmap_paths[(model_label, row_index, stage_index)] = path
                cell_index.append(
                    {
                        "cell_type": "control_residual",
                        "model": model_label,
                        "experiment": SCORE_EXPERIMENT if model_label == "Score" else VIEW_SCORE_EXPERIMENT,
                        "image_id": image_id,
                        "mode": "reconstruction",
                        "block_index": int(block),
                        "stage": STAGE_LABELS[stage_index],
                        "requested_timestep_fraction": float(requested_fraction),
                        "actual_timestep_fraction": actual_fraction,
                        "source_path": str(path) if path else "",
                        "source_exists": bool(path and path.exists()),
                    }
                )

    # Layout: five reference cells, one shared block-label column, then two 5x3 grids.
    header_height = max(66, int(cell_size * 0.28))
    footer_height = max(62, int(cell_size * 0.25))
    reference_width = cell_size + max(24, int(cell_size * 0.10))
    row_label_width = max(118, int(cell_size * 0.50))
    group_gap = max(44, int(cell_size * 0.22))
    grid_height = len(blocks) * cell_size
    score_x = reference_width + row_label_width
    view_score_x = score_x + 3 * cell_size + group_gap
    width = view_score_x + 3 * cell_size
    height = header_height + grid_height + footer_height

    panel = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    stage_font = get_font(max(24, int(cell_size * 0.12)), bold=True)
    block_font = get_font(max(22, int(cell_size * 0.105)), bold=True)
    reference_font = get_font(max(18, int(cell_size * 0.085)), bold=True)
    group_font = get_font(max(26, int(cell_size * 0.13)), bold=True)

    # Stage headings are repeated above the two model grids.
    for group_x in (score_x, view_score_x):
        for stage_index, label in enumerate(STAGE_LABELS):
            x0 = group_x + stage_index * cell_size
            draw_centered(
                draw,
                (x0, 0, x0 + cell_size, header_height),
                label,
                stage_font,
            )

    # Left reference column and the block labels share the five grid rows.
    reference_image_height = cell_size - max(36, int(cell_size * 0.16))
    for row_index, ((reference_label, reference_path), block) in enumerate(zip(references, blocks)):
        y0 = header_height + row_index * cell_size
        reference_image = fit_image(
            reference_path,
            (cell_size, reference_image_height),
        )
        panel.paste(reference_image, (0, y0))
        draw_centered(
            draw,
            (0, y0 + reference_image_height, reference_width, y0 + cell_size),
            reference_label,
            reference_font,
        )
        draw_centered(
            draw,
            (reference_width, y0, reference_width + row_label_width, y0 + cell_size),
            f"Block\n{block}",
            block_font,
        )
        cell_index.append(
            {
                "cell_type": "reference",
                "model": "",
                "experiment": "",
                "image_id": image_id,
                "mode": "reconstruction",
                "block_index": "",
                "stage": reference_label,
                "requested_timestep_fraction": "",
                "actual_timestep_fraction": "",
                "source_path": str(reference_path) if reference_path else "",
                "source_exists": bool(reference_path and reference_path.exists()),
            }
        )

    # Heatmap/overlay grids.
    for model_label, group_x in (("Score", score_x), ("View-Score", view_score_x)):
        for row_index, _block in enumerate(blocks):
            y0 = header_height + row_index * cell_size
            for stage_index, _stage in enumerate(STAGE_LABELS):
                x0 = group_x + stage_index * cell_size
                source = heatmap_paths[(model_label, row_index, stage_index)]
                panel.paste(fit_image(source, (cell_size, cell_size)), (x0, y0))
                draw.rectangle(
                    (x0, y0, x0 + cell_size - 1, y0 + cell_size - 1),
                    outline=(236, 236, 236),
                    width=1,
                )

    draw_centered(
        draw,
        (score_x, header_height + grid_height, score_x + 3 * cell_size, height),
        "Score",
        group_font,
    )
    draw_centered(
        draw,
        (view_score_x, header_height + grid_height, view_score_x + 3 * cell_size, height),
        "View-Score",
        group_font,
    )

    if output_path is None:
        output_dir = ensure_dir(
            output_root / "mechanism_panels" / "score_vs_view_score_reconstruction"
        )
        output_path = output_dir / f"{image_id}__reconstruction__controlnet_blocks.png"
    else:
        output_path = output_path.expanduser()
        ensure_dir(output_path.parent)

    panel.save(output_path, format="PNG")
    index_path = output_path.with_name(output_path.stem + "__cell_index.csv")
    pd.DataFrame(cell_index).to_csv(index_path, index=False, encoding="utf-8-sig")

    missing = sum(not bool(row["source_exists"]) for row in cell_index)
    print(f"[PANEL] saved: {output_path}")
    print(f"[PANEL] cell index: {index_path}")
    print(f"[PANEL] missing source cells: {missing}")
    return output_path, index_path


def validate_user_settings() -> None:
    if RUN_MODE not in {"all", "infer", "panel"}:
        raise ValueError("RUN_MODE must be 'all', 'infer', or 'panel'.")
    if not str(TARGET_IMAGE_NAME).strip():
        raise ValueError("TARGET_IMAGE_NAME cannot be empty.")
    if not RECONSTRUCTION_DIMENSIONS:
        raise ValueError("RECONSTRUCTION_DIMENSIONS cannot be empty.")
    if RENDER_MODE not in {"overlay", "heatmap"}:
        raise ValueError("RENDER_MODE must be 'overlay' or 'heatmap'.")
    if CELL_SIZE < 96:
        raise ValueError("CELL_SIZE must be at least 96.")
    if len(BLOCKS) != 5 or any(block < 0 or block > 12 for block in BLOCKS):
        raise ValueError("BLOCKS must contain five indices between 0 and 12.")
    if len(STAGE_FRACTIONS) != 3 or any(
        fraction < 0.0 or fraction > 1.0 for fraction in STAGE_FRACTIONS
    ):
        raise ValueError("STAGE_FRACTIONS must contain three values between 0 and 1.")
    if RUN_MODE in {"all", "infer"} and not Path(INFERENCE_SCRIPT).expanduser().exists():
        raise FileNotFoundError(
            "INFERENCE_SCRIPT does not exist. Please correct it in USER SETTINGS: "
            f"{INFERENCE_SCRIPT}"
        )


def main() -> None:
    validate_user_settings()
    image_id = image_id_from_name(TARGET_IMAGE_NAME)
    inferred_tags: Dict[str, str] = {}

    if RUN_MODE in ("all", "infer"):
        inferred_tags = run_one_image_inference()

    if RUN_MODE in ("all", "panel"):
        score_tag = SCORE_OUTPUT_TAG or inferred_tags.get(SCORE_EXPERIMENT)
        view_score_tag = VIEW_SCORE_OUTPUT_TAG or inferred_tags.get(VIEW_SCORE_EXPERIMENT)
        build_combined_panel(
            output_root=Path(OUTPUT_ROOT),
            image_id=image_id,
            blocks=BLOCKS,
            stage_fractions=STAGE_FRACTIONS,
            score_tag=score_tag,
            view_score_tag=view_score_tag,
            render_mode=RENDER_MODE,
            cell_size=CELL_SIZE,
            output_path=OUTPUT_PATH,
        )


if __name__ == "__main__":
    main()

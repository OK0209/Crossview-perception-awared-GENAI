#!/usr/bin/env python3
"""一次完成 reconstruction 推理、middle-difference 提取、叠加图；无命令行参数。
只修改顶部 USER SETTINGS。与 figure9_sweep_overlay.py 放在同一目录。
loader、提示词、投影、特征 RMS 沿用用户提供的 20261012-041900 推理版本。
"""
from __future__ import annotations
import sys
import importlib.util
import gc
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import hashlib
import json
import math
import random
import shutil
import types

# ================= USER SETTINGS：只改这里 =================
PROJECT_ROOT=Path("/home/yuebing/GenAI_density_urbanization")
BASE_DATA_DIR=PROJECT_ROOT/"urban_data"
STANDARDIZED_CSV=BASE_DATA_DIR/"GSV_label_data_standardized.csv"
IMAGE_DIR=BASE_DATA_DIR/"images_new"
LAYOUT_DIR=BASE_DATA_DIR/"layout_hint_colored_141_north"
MODEL_CONFIG_PATH=PROJECT_ROOT/"models/cldm_v15_unified.yaml"
CKPT_ROOT=PROJECT_ROOT/"ckpts_s/checkpoints_perception"
OUTPUT_ROOT=BASE_DATA_DIR/"figure9_reconstruction_middle"
FIGURE_OUTPUT_DIR=OUTPUT_ROOT/"figures/level-score"
MODEL_SETTINGS={
    # checkpoint 可填绝对路径；文件名默认在 CKPT_ROOT/<experiment>/multi 下查找。
    # 两个示例沿用参考代码的文件名，请确认各模型确实有对应 checkpoint。
    "View-Level":{"experiment":"view_level_only","checkpoint":"epoch19-step135840.ckpt"},
    "View-Score":{"experiment":"view_only","checkpoint":"epoch19-step135840.ckpt"},
    # 可增加："View-Level":{"experiment":"view_level_only","checkpoint":"具体文件.ckpt"}
}
TARGET_IMAGE_IDS=[
    "514133a2fdc9f04926004394"
]
# 与训练/原推理保持一致；此处沿用本次提供代码的六维及其顺序。
# 使用四维模型时改成训练时实际的四维及顺序，不要混用。
DIMENSIONS=["beautiful","safety","wealthy","boring","depressing","lively"]
SPLIT="all"
IMAGE_RESOLUTION=512
DDIM_STEPS=50
STRENGTH=1.0
GUIDANCE_SCALE=15.0
ETA=0.0
NUM_SAMPLES=1  # 每个 ID 一个样本，与 recorder 的 batch[0] 对齐。
SEED=1111
DEVICE="auto"
A_PROMPT="best quality, extremely detailed, street scene, no watermark, no logo, no text"
N_PROMPT="lowres, bad anatomy, cropped, low quality, blurry, distorted, noise"
APPEND_QUALITY_PROMPT=False
STATIC_TAIL="Follow only static urban features (no people, no vehicles)."
ANALYSIS_TIMESTEP_FRACTIONS=[0.3,0.6,0.9]
OVERWRITE=False  # True 仅重新生成本脚本 OUTPUT_ROOT 中的配套数据。
OVERLAY_ALPHA=0.45
COLORMAP="jet"
NORMALIZATION="per_map"  # per_map 或 shared
FIGURE_DPI=300
REFERENCE_PATHS={"ground_truth":None,"bev":None,"view":None}
# ==========================================================

# 原始 InferenceConfig 的兼容字段，reconstruction 不进行感知等级 sweep。
RUN_MODE="reconstruction"
EXPERIMENT_ORDER=[item["experiment"] for item in MODEL_SETTINGS.values()]
CKPT_NAMES=[item["checkpoint"] for item in MODEL_SETTINGS.values()]
MAX_RECON=None
SWEEP_LEVELS=[]
SWEEP_CONTEXT="row"
SAME_LATENT_PER_SWEEP=True
SAVE_REAL_AND_HINT=True
CONTROL_ANALYSIS_MAX_BLOCKS=0  # 只保存 middle delta，不保存多层 residual。

sys.path.insert(0,str(PROJECT_ROOT))
import cv2
import einops
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm
from cldm.model import create_model, load_state_dict as cldm_load_state_dict
from cldm.ddim_hacked import DDIMSampler

CONTROL_ANALYSIS_SAVE_OVERLAY = True


CONTROL_ANALYSIS_SAVE_OVERLAY_ON_VIEWHINT = True


EXPERIMENTS: Dict[str, Dict[str, bool]] = {
    "baseline": {
        "use_view_transform": False,
        "use_level_prompt": False,
        "use_level_cue_prompt": False,
    },
    "view_only": {
        "use_view_transform": True,
        "use_level_prompt": False,
        "use_level_cue_prompt": False,
    },
    "view_level_only": {
        "use_view_transform": True,
        "use_level_prompt": True,
        "use_level_cue_prompt": False,
    },
    "full": {
        "use_view_transform": True,
        "use_level_prompt": False,
        "use_level_cue_prompt": True,
    },
}


DIM_ATTRS_EN = {
    "beautiful": [
        "greenery",
        "harmony",
        "charm",
    ],
    "safety": [
        "order",
        "visibility",
        "upkeep",
    ],
    "wealthy": [
        "maintenance",
        "spaciousness",
        "investment",
    ],
    "boring": [
        "repetition",
        "monotony",
        "uniformity",
    ],
    "depressing": [
        "harshness",
        "scarcity",
        "decay",
    ],
    "lively": [
        "diversity",
        "activity",
        "variation",
    ],
}


LEVEL_WORDS_5 = {
    1: "very low",
    2: "low",
    3: "medium",
    4: "high",
    5: "very high",
}


LEVEL_TO_SCORE = {
    "very low": 1.0,
    "low": 2.0,
    "medium": 5.0,
    "high": 8.0,
    "very high": 9.0,
}


@dataclass
class InferenceConfig:
    run_mode: str
    experiment_order: Tuple[str, ...]
    dimensions: Tuple[str, ...]
    split: str

    standardized_csv: Path
    image_dir: Path
    layout_dir: Path
    model_config_path: Path
    ckpt_root: Path
    ckpt_names: Tuple[str, ...]
    output_root: Path

    max_recon: Optional[int]
    target_image_ids: Tuple[str, ...]
    sweep_levels: Tuple[str, ...]
    sweep_context: str
    same_latent_per_sweep: bool

    image_resolution: int
    ddim_steps: int
    strength: float
    guidance_scale: float
    eta: float
    num_samples: int
    seed: int
    device: str

    a_prompt: str
    n_prompt: str
    append_quality_prompt: bool
    static_tail: str

    save_real_and_hint: bool
    overwrite: bool


def build_config_from_user_settings() -> InferenceConfig:
    return InferenceConfig(
        run_mode=RUN_MODE,
        experiment_order=tuple(EXPERIMENT_ORDER),
        dimensions=tuple(DIMENSIONS),
        split=SPLIT,
        standardized_csv=Path(STANDARDIZED_CSV),
        image_dir=Path(IMAGE_DIR),
        layout_dir=Path(LAYOUT_DIR),
        model_config_path=Path(MODEL_CONFIG_PATH),
        ckpt_root=Path(CKPT_ROOT),
        ckpt_names=tuple(CKPT_NAMES),
        output_root=Path(OUTPUT_ROOT),
        max_recon=MAX_RECON,
        target_image_ids=tuple(TARGET_IMAGE_IDS),
        sweep_levels=tuple(str(x) for x in SWEEP_LEVELS),
        sweep_context=SWEEP_CONTEXT,
        same_latent_per_sweep=SAME_LATENT_PER_SWEEP,
        image_resolution=IMAGE_RESOLUTION,
        ddim_steps=DDIM_STEPS,
        strength=STRENGTH,
        guidance_scale=GUIDANCE_SCALE,
        eta=ETA,
        num_samples=NUM_SAMPLES,
        seed=SEED,
        device=DEVICE,
        a_prompt=A_PROMPT,
        n_prompt=N_PROMPT,
        append_quality_prompt=APPEND_QUALITY_PROMPT,
        static_tail=STATIC_TAIL,
        save_real_and_hint=SAVE_REAL_AND_HINT,
        overwrite=OVERWRITE,
    )


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_everything(seed, workers=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_int_hash(text: str, modulo: int = 10_000_000) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % modulo


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stem_no_ext(x: str) -> str:
    return Path(str(x)).stem


def find_file(root_dir: Union[str, Path], image_id_or_name: str) -> Optional[Path]:
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


def safe_copy_or_save_image(src_path: Union[str, Path], dst_path: Union[str, Path], resolution: int) -> bool:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    ensure_dir(dst_path.parent)

    if not src_path.exists():
        return False

    try:
        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            shutil.copy2(src_path, dst_path)
            return True

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
        Image.fromarray(img).save(dst_path, quality=95)
        return True
    except Exception:
        shutil.copy2(src_path, dst_path)
        return True


def resize_float_map(arr: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D attention map, got {arr.shape}")
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_CUBIC)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin + 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)


def fraction_to_step_indices(total_steps: int, fractions: List[float]) -> List[Tuple[int, float]]:
    """
    Map analysis fractions to actual DDIM denoising loop indices.

    0.2 / 0.5 / 0.8 are interpreted relative to the actual number of sampler
    steps. With 50 DDIM steps this gives about steps 10 / 25 / 40; with 20
    steps it gives about 4 / 10 / 16. These correspond roughly to early
    structure formation, middle structure-semantic coupling, and late visual
    detail refinement.
    """
    if total_steps <= 0:
        return []
    out = {}
    for frac in fractions:
        f = float(frac)
        idx = int(round(f * max(total_steps - 1, 0)))
        idx = max(0, min(total_steps - 1, idx))
        out[idx] = f
    return [(idx, out[idx]) for idx in sorted(out.keys())]


def heatmap_from_feature_tensor(x: torch.Tensor, image_resolution: int) -> np.ndarray:
    """Channel-norm feature summary used for ControlNet residual/delta maps."""
    if x.ndim != 4:
        raise ValueError(f"Expected feature tensor [B,C,H,W], got {tuple(x.shape)}")
    h = torch.sqrt(torch.mean(x.detach().float()[0] ** 2, dim=0) + 1e-12)
    h = h.unsqueeze(0).unsqueeze(0)
    h = F.interpolate(h, size=(image_resolution, image_resolution), mode="bilinear", align_corners=False)
    return h[0, 0].cpu().numpy().astype(np.float32)


def feature_summary_stats(x: torch.Tensor) -> Dict[str, float]:
    y = x.detach().float()
    return {
        "mean": float(y.mean().cpu().item()),
        "std": float(y.std(unbiased=False).cpu().item()),
        "l1_norm": float(y.abs().mean().cpu().item()),
        "l2_norm": float(torch.sqrt(torch.mean(y * y) + 1e-12).cpu().item()),
        "max_abs": float(y.abs().max().cpu().item()),
    }


def read_rgb_image(path: Union[str, Path], image_resolution: int) -> Optional[np.ndarray]:
    if path is None or not Path(path).exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_resolution, image_resolution), interpolation=cv2.INTER_AREA)
    return img


def save_heatmap_and_overlays(
    heatmap: np.ndarray,
    npy_path: Path,
    png_path: Path,
    image_resolution: int,
    generated_path: Optional[Path] = None,
    view_hint_path: Optional[Path] = None,
) -> Dict[str, str]:
    ensure_dir(png_path.parent)
    h = resize_float_map(heatmap, image_resolution).astype(np.float32)
    np.save(npy_path, h)
    h_u8 = normalize_to_uint8(h)
    heat_bgr = cv2.applyColorMap(h_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(heat_rgb).save(png_path)

    paths = {"npy_path": str(npy_path), "png_path": str(png_path), "overlay_path": "", "overlay_view_hint_path": ""}
    if generated_path is not None and Path(generated_path).exists():
        base = cv2.imread(str(generated_path), cv2.IMREAD_COLOR)
        if base is not None:
            base = cv2.resize(base, (image_resolution, image_resolution), interpolation=cv2.INTER_AREA)
            overlay = cv2.addWeighted(base, 0.60, heat_bgr, 0.40, 0)
            overlay_path = png_path.with_name(png_path.stem.replace("control_", "control_overlay_") + ".png")
            cv2.imwrite(str(overlay_path), overlay)
            paths["overlay_path"] = str(overlay_path)

    if view_hint_path is not None and Path(view_hint_path).exists():
        base = cv2.imread(str(view_hint_path), cv2.IMREAD_COLOR)
        if base is not None:
            base = cv2.resize(base, (image_resolution, image_resolution), interpolation=cv2.INTER_AREA)
            overlay = cv2.addWeighted(base, 0.60, heat_bgr, 0.40, 0)
            overlay_path = png_path.with_name(png_path.stem.replace("control_", "control_viewhint_overlay_") + ".png")
            cv2.imwrite(str(overlay_path), overlay)
            paths["overlay_view_hint_path"] = str(overlay_path)
    return paths


def extract_view_hint_masks(view_hint_path: Union[str, Path], image_resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract approximate road/building/background masks from the analysis view_hint.
    Road is orange-dominant, building is blue-dominant, and white/empty is background.
    """
    img = read_rgb_image(view_hint_path, image_resolution)
    if img is None:
        z = np.zeros((image_resolution, image_resolution), dtype=bool)
        return z, z, ~z
    arr = img.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    nonwhite = np.min(255.0 - arr, axis=2) > 10
    road = nonwhite & (r > b + 25) & (g > b + 5)
    building = nonwhite & (b > r + 20) & (b > g + 10)
    background = ~(road | building)
    return road, building, background


def region_response_stats(heatmap: np.ndarray, view_hint_path: Union[str, Path], image_resolution: int) -> Dict[str, float]:
    h = resize_float_map(heatmap, image_resolution).astype(np.float32)
    h01 = h.astype(np.float32)
    hmin, hmax = float(np.min(h01)), float(np.max(h01))
    if hmax > hmin + 1e-8:
        h01 = (h01 - hmin) / (hmax - hmin)
    else:
        h01 = np.zeros_like(h01)
    road, building, background = extract_view_hint_masks(view_hint_path, image_resolution)

    def masked_mean(mask):
        return float(h01[mask].mean()) if mask.any() else 0.0

    vals = {
        "road_response_mean": masked_mean(road),
        "building_response_mean": masked_mean(building),
        "background_response_mean": masked_mean(background),
    }
    total = vals["road_response_mean"] + vals["building_response_mean"] + vals["background_response_mean"] + 1e-8
    vals["road_response_ratio"] = vals["road_response_mean"] / total
    vals["building_response_ratio"] = vals["building_response_mean"] / total
    vals["background_response_ratio"] = vals["background_response_mean"] / total
    return vals


@torch.no_grad()
def save_view_hint_visualization(model, control: torch.Tensor, cond: Dict, out_path: Path, image_resolution: int) -> str:
    """
    Added in _cross: save the street-view conditioning produced by the model's
    own view transform. FixedNorthBEVProjector.visualize_projection parses the
    BEV color semantics, ray-projects road/building hits, and shades orange/blue
    by normalized depth/visibility. If a future projector lacks that method,
    ControlLDM.visualize_control_projection falls back to projected channel
    intensity as a depth proxy.
    """
    ensure_dir(out_path.parent)
    vis = model.visualize_control_projection(control, cond=cond)
    vis = vis.detach().float().clamp(0.0, 1.0)
    if vis.shape[-2:] != (image_resolution, image_resolution):
        vis = F.interpolate(vis, size=(image_resolution, image_resolution), mode="bilinear", align_corners=False)

    arr = vis[0].permute(1, 2, 0).cpu().numpy()
    # Existing projector visualization uses zero background; the analysis output
    # uses white for no projection, per requested convention.
    empty = np.max(arr, axis=2) <= 1e-6
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    arr[empty] = np.array([255, 255, 255], dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(out_path)
    return str(out_path)


def get_or_save_shared_view_hint(
    model,
    control: torch.Tensor,
    out_base: Path,
    image_id: str,
    use_view_transform: bool,
    image_resolution: int,
    overwrite: bool = False,
) -> Tuple[str, str]:
    """
    Save one reusable projected view_hint per experiment/checkpoint/image.
    The projection depends on the BEV hint and model geometry, not on target_dim
    or level, so all sweep rows for the same image can share this file.
    """
    if not use_view_transform:
        return "", ""

    out_path = out_base / "view_hint" / f"{image_id}_view_hint.png"
    if out_path.exists() and not overwrite:
        return str(out_path), ""

    try:
        p = save_view_hint_visualization(
            model,
            control,
            cond={"use_view_transform": use_view_transform, "control_is_projected": False},
            out_path=out_path,
            image_resolution=image_resolution,
        )
        return p, ""
    except Exception as e:
        return "", str(e)


def detect_image_col(df):
    candidates = [
        "image_file",
        "image_name",
        "image_id",
        "img_id",
        "filename",
        "file_name",
        "name",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find image id column. Tried {candidates}. Existing columns: {list(df.columns)}")


def detect_hint_col(df):
    candidates = [
        "hint_file",
        "layout_file",
        "layout_name",
        "hint_name",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_split_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["split", "dataset_split", "phase", "train_val_test"]:
        if c in df.columns:
            return c
    return None


def load_label_table(
    cfg: InferenceConfig,
) -> Tuple[pd.DataFrame, str]:
    if not cfg.standardized_csv.exists():
        raise FileNotFoundError(
            f"standardized_csv not found: "
            f"{cfg.standardized_csv}"
        )

    df = pd.read_csv(cfg.standardized_csv)



    # --------------------------------------------
    # 1. Filter invalid rows
    # --------------------------------------------
    if "is_valid" in df.columns:
        before = len(df)

        numeric_valid = pd.to_numeric(
            df["is_valid"],
            errors="coerce",
        ).eq(1)

        text_valid = (
            df["is_valid"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "yes"])
        )

        df = df[
            numeric_valid | text_valid
        ].copy()

        print(
            f"[DATA] is_valid filter: "
            f"{before} -> {len(df)}"
        )
    else:
        print(
            "[WARN] No 'is_valid' column found. "
            "No validity filtering was applied."
        )

    # --------------------------------------------
    # 2. Detect file columns
    # --------------------------------------------
    image_col = detect_image_col(df)
    hint_col = detect_hint_col(df)

    df["_image_name_raw"] = (
        df[image_col].astype(str)
    )

    if hint_col is not None:
        df["_hint_name_raw"] = (
            df[hint_col].astype(str)
        )
    else:
        print(
            "[WARN] No hint column found. "
            "Assuming hint and image have "
            "the same filename/stem."
        )
        df["_hint_name_raw"] = (
            df["_image_name_raw"]
        )

    df["_image_id"] = (
        df["_image_name_raw"]
        .map(stem_no_ext)
    )

    # --------------------------------------------
    # 3. Split filter
    # --------------------------------------------
    split_col = detect_split_col(df)

    if cfg.split.lower() != "all":
        if split_col is not None:
            before = len(df)

            df = df[
                df[split_col]
                .astype(str)
                .str.strip()
                .str.lower()
                == cfg.split.lower()
            ].copy()

            print(
                f"[DATA] split={cfg.split}: "
                f"{before} -> {len(df)} "
                f"by column '{split_col}'"
            )
        else:
            print(
                f"[WARN] split={cfg.split} "
                "requested, but no split column "
                "was found. Using all rows."
            )

    # --------------------------------------------
    # 4. Check required perception values
    # --------------------------------------------
    # 只检查指定图片，避免为少量样本扫描全部文件路径。
    if cfg.target_image_ids:
        df = df[df["_image_id"].astype(str).isin(cfg.target_image_ids)].copy()

    required_score_cols = [
        dim for dim in cfg.dimensions
        if dim in df.columns
    ]

    if required_score_cols:
        before = len(df)
        df = df.dropna(
            subset=required_score_cols
        ).copy()

        print(
            "[DATA] non-null perception scores: "
            f"{before} -> {len(df)}"
        )

    # --------------------------------------------
    # 5. Check image and hint paths
    # --------------------------------------------
    valid_rows = []
    missing_image = 0
    missing_hint = 0

    for _, row in df.iterrows():
        image_name = str(
            row["_image_name_raw"]
        )
        hint_name = str(
            row["_hint_name_raw"]
        )
        image_id = str(
            row["_image_id"]
        )

        img_path = (
            find_file(
                cfg.image_dir,
                image_name,
            )
            or find_file(
                cfg.image_dir,
                image_id,
            )
        )

        hint_path = (
            find_file(
                cfg.layout_dir,
                hint_name,
            )
            or find_file(
                cfg.layout_dir,
                stem_no_ext(hint_name),
            )
            or find_file(
                cfg.layout_dir,
                image_id,
            )
        )

        if img_path is None:
            missing_image += 1
            continue

        if hint_path is None:
            missing_hint += 1
            continue

        valid_rows.append(row)

    out = pd.DataFrame(
        valid_rows
    ).reset_index(drop=True)

    print(
        "[DATA] usable rows with "
        f"image + hint: {len(out)}"
    )
    print(
        "[DATA] skipped rows: "
        f"missing_image={missing_image}, "
        f"missing_hint={missing_hint}"
    )

    if len(out) == 0:
        raise RuntimeError(
            "No usable samples found. "
            "Please check CSV columns, "
            "IMAGE_DIR, and LAYOUT_DIR."
        )

    return out, image_col


def normalize_level(value) -> str:
    """
    将 level 转成 prompt 里的文字。

    推荐 CSV 中 level 直接是 low / medium / high。
    如果是数字：
      - 0/1/2 通常解释为 low/medium/high
      - 1/2/3 通常解释为 low/medium/high
      - 1/2/3/4/5 在 sweep 里可解释为 very low ... very high

    注意：单个数字 1/2/3 本身有歧义。因此正式推理最好让 CSV 中的 *_level 保存为字符串。
    """
    if value is None or pd.isna(value):
        return "medium"

    if isinstance(value, str):
        s = value.strip().lower().replace("_", " ").replace("-", " ")
        if s in ["very low", "low", "medium", "high", "very high"]:
            return s
        if s in ["0", "1", "2"]:
            return {"0": "low", "1": "medium", "2": "high"}[s]
        if s in ["3", "4", "5"]:
            return LEVEL_WORDS_5[int(s)]
        return "medium"

    try:
        iv = int(value)
    except Exception:
        return "medium"

    # 这里默认把 0/1/2 作为三分类编码。
    if iv in [0, 1, 2]:
        return {0: "low", 1: "medium", 2: "high"}[iv]

    if iv in LEVEL_WORDS_5:
        return LEVEL_WORDS_5[iv]

    return "medium"


def score_to_level(score) -> str:
    if score is None or pd.isna(score):
        return "medium"
    try:
        x = float(score)
    except Exception:
        return "medium"

    # 只有当 CSV 没有 *_level 时才兜底使用这个粗略映射。
    if x < 3.333:
        return "low"
    if x > 6.667:
        return "high"
    return "medium"


def get_dim_level(row: pd.Series, dim: str) -> str:
    candidate_cols = [
        f"{dim}_cls",
        f"{dim}_level",
    ]

    for col in candidate_cols:
        if col in row.index and pd.notna(row[col]):
            return normalize_level(row[col])

    if dim in row.index and pd.notna(row[dim]):
        print(
            f"[WARN] Missing {dim}_cls; "
            f"falling back to score-based level."
        )
        return score_to_level(row[dim])

    return "medium"


def get_dim_score(row: pd.Series, dim: str) -> float:
    if dim in row.index and pd.notna(row[dim]):
        try:
            return float(row[dim])
        except Exception:
            pass
    lv = get_dim_level(row, dim)
    return float(LEVEL_TO_SCORE.get(lv, 5.0))


def normalize_sweep_level(x: str) -> str:
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    if s in ["very low", "low", "medium", "high", "very high"]:
        return s
    if s in ["1", "2", "3", "4", "5"]:
        return LEVEL_WORDS_5[int(s)]
    if s in ["0"]:
        return "low"
    return "medium"


def quality_suffix(cfg: InferenceConfig) -> str:
    return f", {cfg.a_prompt}" if cfg.append_quality_prompt and cfg.a_prompt else ""


def build_raw_score_prompt(
    row: pd.Series,
    cfg: InferenceConfig,
    override_scores: Optional[Dict[str, float]] = None,
) -> str:
    override_scores = override_scores or {}
    parts = []

    for dim in cfg.dimensions:
        score = override_scores.get(dim, get_dim_score(row, dim))
        parts.append(f"{dim} score of {score:.2f}")

    return (
        "This is a street view with a human perception "
        + ", ".join(parts)
        + ". "
        + cfg.static_tail
        + quality_suffix(cfg)
    )


def build_level_prompt(
    row: pd.Series,
    cfg: InferenceConfig,
    override_levels: Optional[Dict[str, str]] = None,
) -> str:
    override_levels = override_levels or {}
    parts = []

    for dim in cfg.dimensions:
        level = override_levels.get(dim, get_dim_level(row, dim))
        level = normalize_sweep_level(level)
        parts.append(f"a {level} level of {dim} perception")

    return (
        "This street view image shows "
        + ", ".join(parts)
        + ". "
        + cfg.static_tail
        + quality_suffix(cfg)
    )


def get_shuffled_full_dims(
    row: pd.Series,
    cfg: InferenceConfig,
) -> List[str]:
    """
    每张图片使用一个确定性随机顺序。

    同一图片：
    - reconstruction 顺序一致；
    - low/medium/high sweep 顺序一致；
    - 不同 checkpoint 和实验之间顺序一致。

    不同图片之间的六维顺序不同。
    """
    dims = list(cfg.dimensions)

    if "_image_id" in row.index:
        image_key = str(row["_image_id"])
    elif "_image_name_raw" in row.index:
        image_key = str(row["_image_name_raw"])
    else:
        image_key = str(row.name)

    order_seed = (
        cfg.seed
        + stable_int_hash(
            f"full-prompt-order-{image_key}"
        )
    )

    local_rng = random.Random(order_seed)
    local_rng.shuffle(dims)

    return dims


def build_level_cue_prompt(
    row: pd.Series,
    cfg: InferenceConfig,
    override_levels: Optional[
        Dict[str, str]
    ] = None,
) -> str:
    """
    用于 full 模型的六维联合 prompt。

    - 六个维度全部保留；
    - 每个维度三个 cue；
    - 每个 cue 都带 level；
    - 每张图片的维度顺序确定性打乱；
    - reconstruction 和 sweep 共用该函数。
    """
    override_levels = override_levels or {}

    dims = get_shuffled_full_dims(
        row=row,
        cfg=cfg,
    )

    clauses = []

    for dim in dims:
        level = normalize_sweep_level(
            override_levels.get(
                dim,
                get_dim_level(row, dim),
            )
        )

        cues = DIM_ATTRS_EN.get(dim, [])

        if len(cues) >= 3:
            cue_text = (
                f"{level} {cues[0]}, "
                f"{level} {cues[1]}, and "
                f"{level} {cues[2]}"
            )

            clause = (
                f"{level} {dim}, "
                f"with {cue_text}"
            )

        elif len(cues) == 2:
            cue_text = (
                f"{level} {cues[0]} and "
                f"{level} {cues[1]}"
            )

            clause = (
                f"{level} {dim},"
                f"with {cue_text}"
            )

        elif len(cues) == 1:
            clause = (
                f"{level} {dim}, "
                f"with {level} {cues[0]}"
            )

        else:
            clause = (
                f"{level} {dim},"
            )

        clauses.append(clause)

    prompt = (
        "This street view shows "
        + "; ".join(clauses)
        + ". Static urban scene."
    )

    return prompt + quality_suffix(cfg)


def build_prompt_for_experiment(
    row: pd.Series,
    cfg: InferenceConfig,
    exp_name: str,
    override_levels: Optional[Dict[str, str]] = None,
    override_scores: Optional[Dict[str, float]] = None,
) -> str:
    exp_cfg = EXPERIMENTS[exp_name]

    if exp_cfg["use_level_cue_prompt"]:
        return build_level_cue_prompt(row, cfg, override_levels=override_levels)

    if exp_cfg["use_level_prompt"]:
        return build_level_prompt(row, cfg, override_levels=override_levels)

    return build_raw_score_prompt(row, cfg, override_scores=override_scores)


def resolve_ckpt_path(cfg: InferenceConfig, exp_name: str, ckpt_selector: str) -> Path:
    direct = Path(ckpt_selector).expanduser()
    if direct.is_absolute():
        if direct.is_file():
            return direct
        raise FileNotFoundError(f"Checkpoint not found: {direct}")
    exp_dir = cfg.ckpt_root / exp_name / "multi"
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment checkpoint directory does not exist: {exp_dir}")

    if ckpt_selector == "best":
        items = sorted(exp_dir.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not items:
            raise FileNotFoundError(f"No best-*.ckpt found in {exp_dir}")
        return items[0]

    if ckpt_selector == "epoch_latest":
        items = sorted(exp_dir.glob("epoch*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not items:
            raise FileNotFoundError(f"No epoch*.ckpt found in {exp_dir}")
        return items[0]

    p = exp_dir / ckpt_selector
    if p.exists():
        return p

    items = sorted(exp_dir.glob(ckpt_selector), key=lambda x: x.stat().st_mtime, reverse=True)
    if items:
        return items[0]

    raise FileNotFoundError(f"Checkpoint not found: {p}")


def strip_state_dict_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        new_state[nk] = v
    return new_state


def load_checkpoint_state(ckpt_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    ckpt_path = str(ckpt_path)
    try:
        state = cldm_load_state_dict(ckpt_path, location="cpu")
    except Exception:
        raw = torch.load(ckpt_path, map_location="cpu")
        if isinstance(raw, dict) and "state_dict" in raw:
            state = raw["state_dict"]
        else:
            state = raw

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")

    return strip_state_dict_prefixes(state)


def load_model_and_sampler(cfg: InferenceConfig, exp_name: str, ckpt_selector: str, device: torch.device):
    ckpt_path = resolve_ckpt_path(cfg, exp_name, ckpt_selector)
    print(f"[MODEL] {exp_name}: {ckpt_path}")

    model = create_model(str(cfg.model_config_path)).cpu()
    state = load_checkpoint_state(ckpt_path)

    try:
        model.load_state_dict(
            state,
            strict=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "\nCheckpoint is incompatible with "
            "the current model architecture.\n"
            f"Experiment: {exp_name}\n"
            f"Checkpoint: {ckpt_path}\n"
            f"Model config: "
            f"{cfg.model_config_path}\n\n"
            f"Original error:\n{exc}"
        ) from exc


    exp_cfg = EXPERIMENTS[exp_name]
    model.use_view_transform_default = exp_cfg["use_view_transform"]

    model.to(device)
    model.eval()
    sampler = DDIMSampler(model)
    return model, sampler, ckpt_path


class ControlAnalysisRecorder:
    """
    Added in _cross: captures ControlNet residual/control features for hint
    mechanism analysis. Cross-attention explains prompt/token control; these
    residual and middle-delta maps explain where BEV/view-transformed hint
    injects structural constraints. Later comparison should use normalized
    spatial maps, not raw value magnitudes.
    """
    def __init__(self, model, cfg: InferenceConfig, fractions: List[float]):
        self.model = model
        self.cfg = cfg
        self.fractions = [float(x) for x in fractions]
        self.target_steps = set()
        self.target_step_fractions = {}
        self.total_steps = int(cfg.ddim_steps)
        self.current_phase = "off"
        self.current_step_index = None
        self.current_timestep = None
        self.records = []
        self.delta_records = []
        self.errors = []
        self._captured = set()

    def configure_targets(self, total_steps: int):
        self.total_steps = int(total_steps)
        self.target_steps = set()
        self.target_step_fractions = {}
        for step_idx, frac in fraction_to_step_indices(total_steps, self.fractions):
            self.target_steps.add(step_idx)
            self.target_step_fractions[step_idx] = frac

    def set_context(self, sampler, cond_obj, cond_arg, t):
        self.current_phase = "cond" if cond_arg is cond_obj else "uncond"
        self.current_timestep = int(t[0].detach().cpu().item())
        self.current_step_index = None
        if hasattr(sampler, "ddim_timesteps"):
            flipped = list(np.flip(sampler.ddim_timesteps))
            try:
                self.current_step_index = int(flipped.index(self.current_timestep))
            except ValueError:
                self.current_step_index = None

    def clear_context(self):
        self.current_phase = "off"
        self.current_step_index = None
        self.current_timestep = None

    def should_capture(self):
        return self.current_phase == "cond" and self.current_step_index in self.target_steps

    @torch.no_grad()
    def capture(self, x_noisy: torch.Tensor, t: torch.Tensor, cond_arg: Dict):
        if not self.should_capture():
            return
        key = (int(self.current_step_index), int(self.current_timestep))
        if key in self._captured:
            return
        self._captured.add(key)

        try:
            if hasattr(self.model, "extract_base_middle_embedding") and hasattr(self.model, "extract_controlled_middle_embedding"):
                base = self.model.extract_base_middle_embedding(x_noisy, t, cond_arg)
                controlled = self.model.extract_controlled_middle_embedding(x_noisy, t, cond_arg)
                delta = controlled - base
                base_flat = base.detach().float().flatten(1)
                controlled_flat = controlled.detach().float().flatten(1)
                cosine_distance = 1.0 - F.cosine_similarity(base_flat, controlled_flat, dim=1).mean()
                d = {
                    "block_index": -1,
                    "block_name": "control_delta_middle",
                    "ddim_step_index": int(self.current_step_index),
                    "original_timestep_value": int(self.current_timestep),
                    "timestep_fraction": float(self.target_step_fractions.get(self.current_step_index, -1.0)),
                    "total_ddim_steps": int(self.total_steps),
                    "tensor_shape": "x".join(str(x) for x in delta.shape),
                    "heatmap": heatmap_from_feature_tensor(delta, self.cfg.image_resolution),
                    "delta_cosine_distance": float(cosine_distance.detach().cpu().item()),
                }
                d.update({f"delta_{k}": v for k, v in feature_summary_stats(delta).items()})
                d.update({f"base_{k}": v for k, v in feature_summary_stats(base).items()})
                d.update({f"controlled_{k}": v for k, v in feature_summary_stats(controlled).items()})
                self.delta_records.append(d)
        except Exception as e:
            self.errors.append(f"middle delta capture failed at step {key}: {e}")

    def save(
        self,
        out_dir: Path,
        sample_meta: Dict,
        raw_hint_path: Optional[Path],
        view_hint_path: Optional[Path],
        generated_path: Optional[Path],
    ) -> Tuple[List[Dict], Optional[str], List[str]]:
        ensure_dir(out_dir)
        if raw_hint_path is not None and Path(raw_hint_path).exists():
            safe_copy_or_save_image(raw_hint_path, out_dir / "raw_hint.png", self.cfg.image_resolution)
        if view_hint_path is not None and Path(view_hint_path).exists():
            safe_copy_or_save_image(view_hint_path, out_dir / "view_hint.png", self.cfg.image_resolution)

        rows = []
        for rec in self.records:
            frac_tag = f"{float(rec['timestep_fraction']):.2f}".replace(".", "p")
            base = f"control_residual_block_{int(rec['block_index']):02d}_step_{int(rec['ddim_step_index']):03d}_frac_{frac_tag}"
            npy_path = out_dir / f"{base}.npy"
            png_path = out_dir / f"{base}.png"
            saved = save_heatmap_and_overlays(
                rec["heatmap"], npy_path, png_path, self.cfg.image_resolution,
                generated_path=generated_path if CONTROL_ANALYSIS_SAVE_OVERLAY else None,
                view_hint_path=view_hint_path if CONTROL_ANALYSIS_SAVE_OVERLAY_ON_VIEWHINT else None,
            )
            region_stats = region_response_stats(rec["heatmap"], view_hint_path, self.cfg.image_resolution) if view_hint_path else {}
            row = {
                **sample_meta,
                "analysis_type": "control_residual",
                "block_index": rec["block_index"],
                "block_name": rec["block_name"],
                "ddim_step_index": rec["ddim_step_index"],
                "original_timestep_value": rec["original_timestep_value"],
                "timestep_fraction": rec["timestep_fraction"],
                "total_ddim_steps": rec["total_ddim_steps"],
                "tensor_shape": rec["tensor_shape"],
                "control_map_npy_path": saved["npy_path"],
                "control_map_png_path": saved["png_path"],
                "control_overlay_path": saved["overlay_path"],
                "control_overlay_view_hint_path": saved["overlay_view_hint_path"],
                **{k: v for k, v in rec.items() if k in ["mean", "std", "l1_norm", "l2_norm", "max_abs"]},
                **region_stats,
            }
            rows.append(row)

        for rec in self.delta_records:
            frac_tag = f"{float(rec['timestep_fraction']):.2f}".replace(".", "p")
            base = f"control_delta_middle_step_{int(rec['ddim_step_index']):03d}_frac_{frac_tag}"
            npy_path = out_dir / f"{base}.npy"
            png_path = out_dir / f"{base}.png"
            saved = save_heatmap_and_overlays(
                rec["heatmap"], npy_path, png_path, self.cfg.image_resolution,
                generated_path=generated_path if CONTROL_ANALYSIS_SAVE_OVERLAY else None,
                view_hint_path=view_hint_path if CONTROL_ANALYSIS_SAVE_OVERLAY_ON_VIEWHINT else None,
            )
            region_stats = region_response_stats(rec["heatmap"], view_hint_path, self.cfg.image_resolution) if view_hint_path else {}
            row = {
                **sample_meta,
                "analysis_type": "control_delta_middle",
                "block_index": rec["block_index"],
                "block_name": rec["block_name"],
                "ddim_step_index": rec["ddim_step_index"],
                "original_timestep_value": rec["original_timestep_value"],
                "timestep_fraction": rec["timestep_fraction"],
                "total_ddim_steps": rec["total_ddim_steps"],
                "tensor_shape": rec["tensor_shape"],
                "control_map_npy_path": saved["npy_path"],
                "control_map_png_path": saved["png_path"],
                "control_overlay_path": saved["overlay_path"],
                "control_overlay_view_hint_path": saved["overlay_view_hint_path"],
                **{k: v for k, v in rec.items() if k != "heatmap"},
                **region_stats,
            }
            rows.append(row)

        meta_path = out_dir / "control_analysis_metadata.csv"
        pd.DataFrame(rows).to_csv(meta_path, index=False, encoding="utf-8-sig")
        if self.errors:
            (out_dir / "control_analysis_warnings.txt").write_text("\n".join(self.errors), encoding="utf-8")
        return rows, str(meta_path), list(self.errors)


def process_hint_image(
    hint_path: Union[str, Path],
    cfg: InferenceConfig,
    device: torch.device,
) -> torch.Tensor:
    hint = cv2.imread(
        str(hint_path),
        cv2.IMREAD_UNCHANGED,
    )

    if hint is None:
        raise FileNotFoundError(
            f"Cannot read hint image: {hint_path}"
        )

    if hint.ndim == 2:
        hint = cv2.cvtColor(
            hint,
            cv2.COLOR_GRAY2BGR,
        )

    elif hint.ndim == 3:
        if hint.shape[2] == 4:
            transparent_mask = hint[:, :, 3] == 0

            if transparent_mask.any():
                hint[transparent_mask] = [
                    255, 255, 255, 255
                ]

            hint = cv2.cvtColor(
                hint,
                cv2.COLOR_BGRA2BGR,
            )

        elif hint.shape[2] > 4:
            hint = hint[:, :, :3]

    else:
        raise ValueError(
            f"Unexpected hint shape: {hint.shape}"
        )

    hint = cv2.cvtColor(
        hint,
        cv2.COLOR_BGR2RGB,
    )

    hint = cv2.resize(
        hint,
        (
            cfg.image_resolution,
            cfg.image_resolution,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    control = (
        torch.from_numpy(hint)
        .float()
        .to(device)
        / 255.0
    )

    control = control.unsqueeze(0)
    control = einops.rearrange(
        control,
        "b h w c -> b c h w",
    )

    return control


# ==========================================================
# Reconstruction: paired inference + middle recording + plot
# ==========================================================

def load_plot_tools():
    path=Path(__file__).with_name("figure9_sweep_overlay.py")
    if not path.is_file():
        raise FileNotFoundError(f"请将两个脚本放在同一目录，缺少 {path}")
    spec=importlib.util.spec_from_file_location("middle_difference_plot_tools",path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)  # 不会执行 sweep main。
    module.MODELS={label:item["experiment"] for label,item in MODEL_SETTINGS.items()}
    module.CHECKPOINT_TAGS={item["experiment"]:None for item in MODEL_SETTINGS.values()}
    module.STAGES=list(zip(ANALYSIS_TIMESTEP_FRACTIONS,["Early","Middle","Late"]))
    module.MAP_SIZE=IMAGE_RESOLUTION
    module.DPI=FIGURE_DPI
    module.OVERLAY_ALPHA=OVERLAY_ALPHA
    module.COLORMAP=COLORMAP
    module.NORMALIZATION=NORMALIZATION
    module.REFERENCE_PATHS=REFERENCE_PATHS.copy()
    return module


@torch.no_grad()
def sample_reconstruction_with_middle(model,sampler,control,prompt,cfg):
    """与参考 sample_images 使用相同条件/采样过程，增加条件分支的记录。"""
    cond={"c_concat":[control],"c_crossattn":[model.get_learned_conditioning([prompt])]}
    un_cond={"c_concat":[control],"c_crossattn":[model.get_learned_conditioning([cfg.n_prompt])]}
    model.control_scales=[cfg.strength]*13
    recorder=ControlAnalysisRecorder(model,cfg,ANALYSIS_TIMESTEP_FRACTIONS)
    recorder.configure_targets(cfg.ddim_steps)
    original_apply=model.apply_model

    def wrapped_apply(x_noisy,t,cond_arg,*more,**options):
        recorder.set_context(sampler,cond,cond_arg,t)
        try:
            recorder.capture(x_noisy,t,cond_arg)
            return original_apply(x_noisy,t,cond_arg,*more,**options)
        finally:
            recorder.clear_context()

    try:
        model.apply_model=wrapped_apply
        samples,_=sampler.sample(cfg.ddim_steps,1,
            (4,cfg.image_resolution//8,cfg.image_resolution//8),cond,
            eta=cfg.eta,unconditional_guidance_scale=cfg.guidance_scale,
            unconditional_conditioning=un_cond)
    finally:
        model.apply_model=original_apply
    image=model.decode_first_stage(samples)
    image=einops.rearrange(image,"b c h w -> b h w c")
    image=(image*127.5+127.5).detach().cpu().numpy()
    return np.clip(image,0,255).astype(np.uint8),recorder


def verify_middle_metadata(path):
    df=pd.read_csv(path)
    middle=df[df["analysis_type"]=="control_delta_middle"].copy()
    fractions=pd.to_numeric(middle["timestep_fraction"],errors="raise")
    for fraction in ANALYSIS_TIMESTEP_FRACTIONS:
        selected=middle[(fractions-fraction).abs()<.001]
        if len(selected)!=1 or not Path(str(selected.iloc[0]["control_map_npy_path"])).is_file():
            raise RuntimeError(f"中间 map 缺失或不唯一：{fraction}，{path}")
    return middle


def input_paths(row,cfg):
    image_id=str(row["_image_id"])
    image=find_file(cfg.image_dir,str(row["_image_name_raw"])) or find_file(cfg.image_dir,image_id)
    hint=(find_file(cfg.layout_dir,str(row["_hint_name_raw"]))
          or find_file(cfg.layout_dir,stem_no_ext(str(row["_hint_name_raw"])))
          or find_file(cfg.layout_dir,image_id))
    if image is None or hint is None:
        raise FileNotFoundError(f"{image_id}：原图或 BEV 不存在")
    return Path(image).resolve(),Path(hint).resolve()


def regenerate_single_overlays(middle,plot_tools,bounds_by_path=None):
    """按当前绘图配置保存独立 overlay，包含缓存复用时的样式更新。"""
    for _,row in middle.iterrows():
        path=Path(row["control_map_npy_path"])
        arr=plot_tools.load_map(path)
        bounds=(bounds_by_path or {}).get(str(path),(float(arr.min()),float(arr.max())))
        image=plot_tools.load_image(Path(row["generated_path"]))
        overlay=plot_tools.overlay_image(image,arr,bounds)
        target=path.with_name(path.stem.replace("control_","control_overlay_")+".png")
        Image.fromarray(overlay).save(target)


def main():
    if not TARGET_IMAGE_IDS or not MODEL_SETTINGS:
        raise ValueError("TARGET_IMAGE_IDS 和 MODEL_SETTINGS 不能为空")
    experiments=[item["experiment"] for item in MODEL_SETTINGS.values()]
    if len(set(experiments))!=len(experiments):
        raise ValueError("每个实验只配置一个 checkpoint；同实验不同 checkpoint 请分别运行")
    if any(exp not in EXPERIMENTS for exp in experiments):
        raise ValueError(f"模型实验名仅支持 {list(EXPERIMENTS)}")
    if NUM_SAMPLES!=1:
        raise ValueError("为确保 map 与图片一一对应，NUM_SAMPLES 必须为 1")
    if len(ANALYSIS_TIMESTEP_FRACTIONS)!=3 or 0.6 not in ANALYSIS_TIMESTEP_FRACTIONS:
        raise ValueError("该布局使用三个采样阶段，且必须包含 0.6；默认 [0.3,0.6,0.9]")
    if (ANALYSIS_TIMESTEP_FRACTIONS != sorted(set(ANALYSIS_TIMESTEP_FRACTIONS))
            or any(not 0 <= fraction <= 1 for fraction in ANALYSIS_TIMESTEP_FRACTIONS)
            or len(fraction_to_step_indices(DDIM_STEPS,ANALYSIS_TIMESTEP_FRACTIONS)) != 3):
        raise ValueError("三个时间阶段必须递增、处于 [0,1]，且对应三个不同采样步")
    if IMAGE_RESOLUTION <= 0 or IMAGE_RESOLUTION % 8:
        raise ValueError("IMAGE_RESOLUTION 必须为正数且能被 8 整除")
    if not MODEL_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"MODEL_CONFIG_PATH 不存在：{MODEL_CONFIG_PATH}")
    ids=list(dict.fromkeys(str(value) for value in TARGET_IMAGE_IDS))
    if any(not value or '/' in value or '\\' in value or value in {'.','..'} for value in ids):
        raise ValueError("TARGET_IMAGE_IDS 应填写纯图片 ID")
    plot_tools=load_plot_tools()
    cfg=build_config_from_user_settings()
    # 开始 GPU 推理前，检查所有 checkpoint，避免先跑完一组才发现路径错误。
    for item in MODEL_SETTINGS.values():
        resolve_ckpt_path(cfg,item["experiment"],item["checkpoint"])
    data,_=load_label_table(cfg)
    if data["_image_id"].astype(str).duplicated().any():
        raise ValueError("CSV 中指定图片 ID 有重复，请先处理")
    row_lookup={str(row["_image_id"]):row for _,row in data.iterrows()}
    missing=[image_id for image_id in ids if image_id not in row_lookup]
    if missing:
        raise RuntimeError(f"指定 ID 未出现在当前 CSV/split 或原图/BEV 缺失：{missing}")
    inputs={image_id:input_paths(row_lookup[image_id],cfg) for image_id in ids}
    root=ensure_dir(OUTPUT_ROOT/"reconstruction").resolve()
    real_dir=ensure_dir(root/"real");bev_dir=ensure_dir(root/"BEV")
    device=get_device(cfg.device)
    records=[];status=[];complete={image_id:set() for image_id in ids}
    resolved_checkpoints={}
    for label,item in MODEL_SETTINGS.items():
        exp=item["experiment"]
        model,sampler,checkpoint=load_model_and_sampler(cfg,exp,item["checkpoint"],device)
        try:
            if not all(callable(getattr(model,name,None)) for name in
                       ("extract_base_middle_embedding","extract_controlled_middle_embedding")):
                raise RuntimeError("模型缺少 middle 特征提取函数，请使用提供的 cldm_unified 版本")
            tag=Path(checkpoint).stem
            resolved_checkpoints[exp]=tag
            generated_dir=ensure_dir(root/"generated"/exp/tag)
            for image_id in ids:
                try:
                    row=row_lookup[image_id]
                    image,hint=inputs[image_id]
                    output_image=generated_dir/f"{image_id}.png"
                    map_dir=root/"control_analysis"/exp/tag/image_id
                    meta_path=map_dir/"control_analysis_metadata.csv"
                    settings_path=map_dir/"sampling_settings.json"
                    real=real_dir/f"{image_id}.png";bev=bev_dir/f"{image_id}.png"
                    # 不传 override_scores / override_levels：使用样本原始多维条件。
                    prompt=build_prompt_for_experiment(row,cfg,exp)
                    seed=cfg.seed+stable_int_hash(f"recon-{image_id}")
                    sampling_config=asdict(cfg);sampling_config.pop("overwrite",None)
                    sampling_settings={"sampling_config":sampling_config,"model":label,"experiment":exp,
                        "image_id":image_id,"prompt":prompt,"sample_seed":seed,
                        "checkpoint":str(checkpoint),"checkpoint_mtime_ns":Path(checkpoint).stat().st_mtime_ns,
                        "hint_source":str(hint),"hint_mtime_ns":hint.stat().st_mtime_ns,
                        "stages":ANALYSIS_TIMESTEP_FRACTIONS}
                    sampling_settings=json.loads(json.dumps(sampling_settings,default=str))
                    cached=output_image.is_file() and meta_path.is_file() and settings_path.is_file()
                    if cached and not OVERWRITE:
                        if json.loads(settings_path.read_text())!=sampling_settings:
                            raise RuntimeError("已有采样设置不同；换 OUTPUT_ROOT 或设 OVERWRITE=True")
                        middle=verify_middle_metadata(meta_path)
                        for column in ["generated_path","real_path","bev_hint_path"]:
                            if not Path(str(middle.iloc[0][column])).is_file():
                                raise FileNotFoundError(f"缓存引用的 {column} 不存在")
                        if EXPERIMENTS[exp]["use_view_transform"]:
                            if not Path(str(middle.iloc[0]["view_hint_path"])).is_file():
                                raise FileNotFoundError("缓存投影 view hint 不存在")
                        meta=middle.iloc[0].to_dict()
                        cache_status="reused"
                    else:
                        if (output_image.exists() or meta_path.exists()) and not OVERWRITE:
                            raise RuntimeError("已有输出不完整；换 OUTPUT_ROOT 或设 OVERWRITE=True")
                        ensure_dir(map_dir)
                        safe_copy_or_save_image(image,real,cfg.image_resolution)
                        safe_copy_or_save_image(hint,bev,cfg.image_resolution)
                        control=process_hint_image(hint,cfg,device)
                        view_path,error=get_or_save_shared_view_hint(model,control,
                            root/"projected"/exp/tag,image_id,
                            EXPERIMENTS[exp]["use_view_transform"],cfg.image_resolution,OVERWRITE)
                        if error:
                            raise RuntimeError(f"投影 view 保存失败：{error}")
                        meta={"mode":"reconstruction","experiment":exp,"ckpt_tag":tag,
                            "ckpt_path":str(checkpoint),"image_id":image_id,
                            "target_dim":"reconstruction","target_level":"original",
                            "generated_path":str(output_image),"real_path":str(real),
                            "bev_hint_path":str(bev),"raw_hint_path":str(hint),"view_hint_path":view_path,
                            "prompt":prompt,"sample_seed":seed}
                        for dimension in cfg.dimensions:
                            meta[f"{dimension}_score"]=get_dim_score(row,dimension)
                            meta[f"{dimension}_level"]=get_dim_level(row,dimension)
                        set_global_seed(seed)
                        images,recorder=sample_reconstruction_with_middle(model,sampler,control,prompt,cfg)
                        if recorder.errors:
                            raise RuntimeError(f"特征记录失败：{recorder.errors}")
                        # 先保存生成图，再保存 recorder，保证单张叠加图也能生成。
                        Image.fromarray(images[0]).save(output_image)
                        _,saved_metadata,errors=recorder.save(map_dir,meta,hint,
                            Path(view_path) if view_path else None,output_image)
                        if errors or saved_metadata is None:
                            raise RuntimeError(f"中间 map 保存失败：{errors}")
                        middle=verify_middle_metadata(Path(saved_metadata))
                        (map_dir/"original_prompt.txt").write_text(prompt,encoding="utf-8")
                        settings_path.write_text(json.dumps(sampling_settings,indent=2),encoding="utf-8")
                        cache_status="generated"
                    records.append({**meta,"model_label":label,"control_analysis_metadata_path":str(meta_path)})
                    status.append({"stage":"inference","image_id":image_id,"model":label,
                                   "status":cache_status,"error":"","output":str(output_image)})
                    complete[image_id].add(exp)
                    print(f"[{cache_status.upper()}] {label}/{image_id}",flush=True)
                except Exception as exc:
                    status.append({"stage":"inference","image_id":image_id,"model":label,
                                   "status":"failed","error":str(exc),"output":""})
                    print(f"[FAILED] {label}/{image_id}: {exc}",flush=True)
        finally:
            del model,sampler
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(root/"metadata_reconstruction_selected.csv",index=False)
    plot_tools.CHECKPOINT_TAGS=resolved_checkpoints
    # 同一脚本最后自动出图，无需再运行任何绘图命令。
    for image_id in ids:
        if complete[image_id]!=set(experiments):
            continue
        try:
            path=plot_tools.render(root,FIGURE_OUTPUT_DIR.resolve(),image_id,
                                  "reconstruction",["original"],"original","reconstruction")
            # 独立 overlay 与组合图使用同样的色图、透明度和归一化范围。
            source_table=pd.read_csv(path.with_name(path.stem+"_sources.csv"))
            bounds_by_path={str(row.map_npy_path):(row.display_min,row.display_max)
                            for row in source_table.itertuples()}
            for item in MODEL_SETTINGS.values():
                metadata=root/"control_analysis"/item["experiment"]/resolved_checkpoints[item["experiment"]]/image_id/"control_analysis_metadata.csv"
                regenerate_single_overlays(verify_middle_metadata(metadata),plot_tools,bounds_by_path)
            status.append({"stage":"figure","image_id":image_id,"model":"comparison",
                           "status":"ok","error":"","output":str(path)})
        except Exception as exc:
            status.append({"stage":"figure","image_id":image_id,"model":"comparison",
                           "status":"failed","error":str(exc),"output":""})
            print(f"[FIGURE FAILED] {image_id}: {exc}",flush=True)
    table=pd.DataFrame(status)
    table.to_csv(OUTPUT_ROOT/"batch_status.csv",index=False)
    print(f"[OUTPUT] paired reconstruction: {root}")
    print(f"[OUTPUT] figures: {FIGURE_OUTPUT_DIR}")
    if (table["status"]=="failed").any():
        raise SystemExit("部分处理失败，请查看 OUTPUT_ROOT/batch_status.csv")


if __name__=="__main__":
    main()



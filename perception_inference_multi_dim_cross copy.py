# -*- coding: utf-8 -*-


from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
from ldm.modules.attention import default, exists


# ============================================================
# 0. USER SETTINGS：只改这里即可
# ============================================================

# ---------- What to run ----------
# 可选："reconstruction" / "sweep" / "both"
RUN_MODE = "sweep"

# 可选模型组："baseline", "view_only", "view_level_only", "full"
EXPERIMENT_ORDER = [
"baseline"
]

# 六个 perception 维度。顺序会影响 prompt 中维度排列。
DIMENSIONS = ["beautiful", "safety", "wealthy", "boring", "depressing", "lively"]

# 推理使用哪个 split。
# 如果 CSV 中有 split 列，则只取对应 split；如果没有 split 列，会自动使用全部样本。
# 可选："train" / "val" / "test" / "all"
SPLIT = "all"


# ---------- Paths ----------
BASE_DATA_DIR = Path("/home/yuebing/GenAI_density_urbanization/urban_data")
STANDARDIZED_CSV = BASE_DATA_DIR / "GSV_label_data_standardized.csv"
IMAGE_DIR = BASE_DATA_DIR / "images_new"
LAYOUT_DIR = BASE_DATA_DIR / "layout_hint_colored_141_north"

MODEL_CONFIG_PATH = Path("./models/cldm_v15_unified.yaml")
CKPT_ROOT = Path("/home/yuebing/GenAI_density_urbanization/ckpts_s/checkpoints_perception")

# 可选：
#   "last.ckpt"               : 使用 last.ckpt
#   "best"                    : 自动选择最新修改的 best-*.ckpt
#   "epoch_latest"            : 自动选择最新修改的 epoch*.ckpt
#   "epoch29-step203760.ckpt" : 使用指定 epoch 文件
#   也可以同时给多个，用于同一实验组分别跑多个 checkpoint。
CKPT_NAMES = [
    "epoch19-step135840.ckpt",
]

OUTPUT_ROOT = BASE_DATA_DIR / "inference_outputs_multidim_ablation/pre"


# ---------- Reconstruction settings ----------
# reconstruction 是从 SPLIT 中抽样做重建；设为 None 或 <=0 表示使用全部可用样本。
MAX_RECON = 1000


# ---------- Sweep settings ----------
# 这里是 sweep 固定使用的图片。
# 也就是说：sweep 不再随机抽 max_sweep_images，而是只对下面这些 image_id 做固定对比。
TARGET_IMAGE_IDS = [
    "513e6146fdc9f0358700b50d",
    "51421a69fdc9f04926008339",
    "513d3836fdc9f03587002c41",
    "50f5e640fdc9f065f0006bb6"
]

# 如果你想做三分类 controllability，就用 low/medium/high。
# 如果你想做五档 sweep，可以改为 ["1", "2", "3", "4", "5"]。
SWEEP_LEVELS = ["low", "medium", "high"]

# 非目标维度怎么处理：
#   "medium" : 非目标维度全部固定为 medium，最适合做单维 controllability 对比。
#   "row"    : 非目标维度使用该图片在 CSV 里的原始标签。
SWEEP_CONTEXT = "medium"

# 对同一个 image_id + target_dim 的不同 level 使用相同 latent seed。
# 这样图像变化更主要来自 prompt level，而不是随机噪声。
SAME_LATENT_PER_SWEEP = True


# ---------- Sampling settings ----------
IMAGE_RESOLUTION = 512
DDIM_STEPS = 50
STRENGTH = 1.0
GUIDANCE_SCALE = 15.0
ETA = 0.0
NUM_SAMPLES = 1
SEED = 1111
DEVICE = "auto"  # "auto" / "cuda:0" / "cpu"

A_PROMPT = "best quality, extremely detailed, street scene, no watermark, no logo, no text"
N_PROMPT = "lowres, bad anatomy, cropped, low quality, blurry, distorted, noise"
APPEND_QUALITY_PROMPT = False
STATIC_TAIL = "Follow only static urban features (no people, no vehicles)."

# 是否保存 real / hint 到 reconstruction 输出文件夹
SAVE_REAL_AND_HINT = True

# 如果 False，遇到已经生成的图片会跳过。
OVERWRITE = False


# ============================================================
# 0X. CROSS ANALYSIS SETTINGS
# ============================================================
# Added in this _cross script only. Disabling these flags keeps sampling as
# close as possible to perception_inference_multi_dim.py.
SAVE_VIEW_HINT = True
SAVE_CROSS_ATTENTION = True
SAVE_CONTROL_ANALYSIS = True
ANALYSIS_TIMESTEP_FRACTIONS = [0.3, 0.6, 0.9]
MAX_ANALYSIS_IMAGES = None

CROSS_ATTENTION_EXPERIMENTS = ["baseline", "full", "view_level_only", "view_only"]
CROSS_ATTENTION_TARGET_DIMS = ["beautiful", "safety", "lively","wealthy","boring","depressing"]
CROSS_ATTENTION_LEVELS = ["low", "medium", "high"]
CROSS_ATTENTION_MAX_IMAGES = None
CROSS_ATTENTION_SAVE_AVG_HEADS = True
CROSS_ATTENTION_SAVE_TOKEN_MAPS = True
CROSS_ATTENTION_SAVE_NUMPY = True
CROSS_ATTENTION_SAVE_PNG = True
CROSS_ATTENTION_SAVE_OVERLAY = True

CONTROL_ANALYSIS_EXPERIMENTS = ["baseline", "full", "view_only", "view_level_only"]
CONTROL_ANALYSIS_TARGET_DIMS = ["beautiful", "safety", "lively","wealthy","boring","depressing"]
CONTROL_ANALYSIS_LEVELS = ["low", "medium", "high"]
CONTROL_ANALYSIS_MAX_BLOCKS = None  # e.g. 13 for all ControlNet outputs; smaller values reduce disk use.
CONTROL_ANALYSIS_SAVE_NUMPY = True
CONTROL_ANALYSIS_SAVE_PNG = True
CONTROL_ANALYSIS_SAVE_OVERLAY = True
CONTROL_ANALYSIS_SAVE_OVERLAY_ON_VIEWHINT = True

CROSS_ATTENTION_KEYWORDS = {
    "beautiful": [
        "beautiful",
        "greenery",
        "harmony",
        "charm",
    ],
    "safety": [
        "safety",
        "order",
        "visibility",
        "upkeep",
    ],
    "wealthy": [
        "wealthy",
        "maintenance",
        "spaciousness",
        "investment",
    ],
    "boring": [
        "boring",
        "repetition",
        "monotony",
        "uniformity",
    ],
    "depressing": [
        "depressing",
        "harshness",
        "scarcity",
        "decay",
    ],
    "lively": [
        "lively",
        "diversity",
        "activity",
        "variation",
    ],
}


# ============================================================
# 1. Experiment definition
# ============================================================

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

# 请与你训练侧 perception_dataset_multidim.py 中的 cue 关键词保持一致。
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


# ============================================================
# 2. Utilities
# ============================================================

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


def sanitize_token_for_filename(token: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(token).strip())
    cleaned = cleaned.strip("_")
    return cleaned[:40] if cleaned else "token"


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


def save_attention_png(attn_map: np.ndarray, out_path: Path, image_resolution: int, overlay_path: Optional[Path] = None) -> None:
    ensure_dir(out_path.parent)
    attn_resized = resize_float_map(attn_map, image_resolution)
    heat_u8 = normalize_to_uint8(attn_resized)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(heat_rgb).save(out_path)

    if overlay_path is not None and overlay_path.exists():
        base = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if base is not None:
            base = cv2.resize(base, (image_resolution, image_resolution), interpolation=cv2.INTER_AREA)
            overlay = cv2.addWeighted(base, 0.60, heat_bgr, 0.40, 0)
            overlay_out = out_path.with_name(out_path.stem.replace("attn_", "attn_overlay_") + ".png")
            cv2.imwrite(str(overlay_out), overlay)


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


def should_capture_cross_attention(exp_name: str, target_dim: str, level_word: str, image_index: int) -> bool:
    if not SAVE_CROSS_ATTENTION:
        return False
    if exp_name not in CROSS_ATTENTION_EXPERIMENTS:
        return False
    if not CROSS_ATTENTION_SAVE_TOKEN_MAPS:
        return False
    if target_dim not in CROSS_ATTENTION_TARGET_DIMS:
        return False
    if normalize_sweep_level(level_word) not in [normalize_sweep_level(x) for x in CROSS_ATTENTION_LEVELS]:
        return False
    limit = CROSS_ATTENTION_MAX_IMAGES if CROSS_ATTENTION_MAX_IMAGES is not None else MAX_ANALYSIS_IMAGES
    if limit is not None and image_index >= int(limit):
        return False
    return True


def should_capture_control_analysis(exp_name: str, target_dim: str, level_word: str, image_index: int) -> bool:
    if not SAVE_CONTROL_ANALYSIS:
        return False
    if exp_name not in CONTROL_ANALYSIS_EXPERIMENTS:
        return False
    if target_dim not in CONTROL_ANALYSIS_TARGET_DIMS:
        return False
    if normalize_sweep_level(level_word) not in [normalize_sweep_level(x) for x in CONTROL_ANALYSIS_LEVELS]:
        return False
    if MAX_ANALYSIS_IMAGES is not None and image_index >= int(MAX_ANALYSIS_IMAGES):
        return False
    return True


# ============================================================
# 3. Data loading
# ============================================================

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


def select_reconstruction_rows(
    df: pd.DataFrame,
    cfg: InferenceConfig,
) -> pd.DataFrame:
    """
    Reconstruction 优先使用 TARGET_IMAGE_IDS 中指定的图片。

    当 TARGET_IMAGE_IDS 为空时，才恢复原来的 MAX_RECON 随机抽样逻辑。
    """

    # --------------------------------------------------------
    # 1. 指定 ID reconstruction
    # --------------------------------------------------------
    if cfg.target_image_ids:
        df2 = df.copy()
        df2["_image_id_str"] = df2["_image_id"].astype(str)

        # image_id -> row
        id_to_row = {
            str(row["_image_id_str"]): row
            for _, row in df2.iterrows()
        }

        selected_rows = []
        missing_ids = []

        for image_id in cfg.target_image_ids:
            image_id = str(image_id)

            if image_id in id_to_row:
                selected_rows.append(id_to_row[image_id])
            else:
                missing_ids.append(image_id)

        if missing_ids:
            print(
                "[WARN] The following reconstruction IDs were not found "
                "in the current CSV/split, or their image/hint is missing:"
            )
            for image_id in missing_ids:
                print(f"       - {image_id}")

        if len(selected_rows) == 0:
            raise RuntimeError(
                "None of TARGET_IMAGE_IDS can be used for reconstruction. "
                "Please check SPLIT, STANDARDIZED_CSV, IMAGE_DIR, and LAYOUT_DIR."
            )

        output = pd.DataFrame(selected_rows).reset_index(drop=True)

        print(
            f"[RECON DATA] specified reconstruction images usable: "
            f"{len(output)} / {len(cfg.target_image_ids)}"
        )

        return output

    # --------------------------------------------------------
    # 2. TARGET_IMAGE_IDS 为空时，使用原随机抽样逻辑
    # --------------------------------------------------------
    if (
        cfg.max_recon is None
        or cfg.max_recon <= 0
        or cfg.max_recon >= len(df)
    ):
        return df.copy().reset_index(drop=True)

    return df.sample(
        n=cfg.max_recon,
        random_state=cfg.seed,
    ).reset_index(drop=True)


def select_sweep_rows(df: pd.DataFrame, cfg: InferenceConfig) -> pd.DataFrame:
    """
    Sweep 使用固定 TARGET_IMAGE_IDS。
    顺序也按照 TARGET_IMAGE_IDS 中的顺序保存，便于后续做图。
    """
    if not cfg.target_image_ids:
        raise ValueError(
            "TARGET_IMAGE_IDS is empty. For fixed sweep, please fill TARGET_IMAGE_IDS in USER SETTINGS."
        )

    df2 = df.copy()
    df2["_image_id_str"] = df2["_image_id"].astype(str)
    id_to_row = {str(r["_image_id_str"]): r for _, r in df2.iterrows()}

    selected = []
    missing = []
    for img_id in cfg.target_image_ids:
        img_id = str(img_id)
        if img_id in id_to_row:
            selected.append(id_to_row[img_id])
        else:
            missing.append(img_id)

    if missing:
        print(f"[WARN] {len(missing)} TARGET_IMAGE_IDS not found in current CSV/split or missing image/hint:")
        for x in missing[:30]:
            print(f"       - {x}")
        if len(missing) > 30:
            print("       ...")

    out = pd.DataFrame(selected).reset_index(drop=True)
    if len(out) == 0:
        raise RuntimeError("None of TARGET_IMAGE_IDS can be used. Check SPLIT, CSV, IMAGE_DIR, and LAYOUT_DIR.")

    print(f"[SWEEP DATA] fixed TARGET_IMAGE_IDS usable: {len(out)} / {len(cfg.target_image_ids)}")
    return out


# ============================================================
# 4. Level / score and prompt builders
# ============================================================

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


def sweep_level_to_score(x: str) -> float:
    lv = normalize_sweep_level(x)
    return float(LEVEL_TO_SCORE.get(lv, 5.0))


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


def build_level_cue_prompt_for_target_dim(
    row: pd.Series,
    cfg: InferenceConfig,
    target_dim: str,
    override_levels: Optional[
        Dict[str, str]
    ] = None,
    include_other_dim_targets: bool = True,
    include_other_dim_cues: bool = False,
) -> str:
    """
    保留现有函数接口，但内部使用与训练和
    reconstruction 完全相同的六维联合 prompt。

    target_dim 的 low/medium/high 已经包含在
    override_levels 中。
    """
    return build_level_cue_prompt(
        row=row,
        cfg=cfg,
        override_levels=override_levels,
    )

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


def build_sweep_overrides(
    row: pd.Series,
    cfg: InferenceConfig,
    target_dim: str,
    target_level: str,
    exp_name: str,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    exp_cfg = EXPERIMENTS[exp_name]
    target_level = normalize_sweep_level(target_level)

    override_levels: Dict[str, str] = {}
    override_scores: Dict[str, float] = {}

    if exp_cfg["use_level_prompt"] or exp_cfg["use_level_cue_prompt"]:
        for dim in cfg.dimensions:
            if dim == target_dim:
                override_levels[dim] = target_level
            else:
                override_levels[dim] = get_dim_level(row, dim) if cfg.sweep_context == "row" else "medium"
    else:
        for dim in cfg.dimensions:
            if dim == target_dim:
                override_scores[dim] = sweep_level_to_score(target_level)
            else:
                override_scores[dim] = get_dim_score(row, dim) if cfg.sweep_context == "row" else 5.0

    return override_levels, override_scores


# ============================================================
# 5. Model loading
# ============================================================

def resolve_ckpt_path(cfg: InferenceConfig, exp_name: str, ckpt_selector: str) -> Path:
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


# ============================================================
# 6. Cross-attention analysis helpers
# ============================================================

def get_clip_token_records(model, prompt: str) -> Dict:
    tokenizer = getattr(getattr(model, "cond_stage_model", None), "tokenizer", None)
    max_length = getattr(getattr(model, "cond_stage_model", None), "max_length", 77)
    if tokenizer is None:
        return {"prompt": prompt, "token_ids": [], "tokens": [], "warning": "Tokenizer not found on cond_stage_model."}

    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return {
        "prompt": prompt,
        "token_ids": token_ids,
        "tokens": [
            {"index": int(i), "token_id": int(tid), "token": str(tok), "decoded": tokenizer.decode([int(tid)])}
            for i, (tid, tok) in enumerate(zip(token_ids, tokens))
        ],
    }


def _keyword_token_ids(model, keyword: str) -> List[int]:
    tokenizer = getattr(getattr(model, "cond_stage_model", None), "tokenizer", None)
    if tokenizer is None:
        return []
    try:
        ids = tokenizer(keyword, add_special_tokens=False)["input_ids"]
    except TypeError:
        ids = tokenizer(keyword)["input_ids"]
        special = set(getattr(tokenizer, "all_special_ids", []))
        ids = [x for x in ids if x not in special]
    return [int(x) for x in ids]


def _add_phrase_token_matches(
    selected: Dict[int, Dict],
    token_ids: List[int],
    token_rows: List[Dict],
    model,
    phrase: str,
    matched_keyword: str,
) -> None:
    ids = _keyword_token_ids(model, phrase) if model is not None else []
    if not ids:
        return
    n = len(ids)
    for start in range(0, max(0, len(token_ids) - n + 1)):
        if token_ids[start:start + n] == ids:
            for off in range(n):
                idx = start + off
                selected[idx] = {
                    "index": idx,
                    "token_id": int(token_ids[idx]),
                    "token": token_rows[idx]["token"],
                    "decoded": token_rows[idx]["decoded"],
                    "matched_keyword": matched_keyword,
                }


def _is_phrase_match(matched_keyword: str) -> bool:
    matched = str(matched_keyword)
    return matched.startswith(("target_score_phrase:", "target_level_phrase:", "cue_phrase:"))


def _preserve_phrase_selection(selected: Dict[int, Dict], idx: int) -> bool:
    matched = str(selected.get(idx, {}).get("matched_keyword", ""))
    return _is_phrase_match(matched)


def select_token_indices(
    token_info: Dict,
    prompt: str,
    target_dim: str,
    target_level: str,
    keywords: Dict,
    model=None,
    target_score: Optional[float] = None,
) -> List[Dict]:
    """
    Added in _cross: choose prompt tokens for later spatial attention maps.
    Phrase matching is done at tokenizer-id level when possible, so split CLIP
    sub-tokens are retained individually; substring matching is a fallback.
    """
    token_ids = token_info.get("token_ids", [])
    token_rows = token_info.get("tokens", [])
    wanted = [target_dim]
    wanted.extend(keywords.get(target_dim, []))

    wanted = [
        str(x).lower()
        for x in wanted
        if str(x).strip()
    ]

    selected: Dict[int, Dict] = {}
    special_ids = set()
    tokenizer = getattr(getattr(model, "cond_stage_model", None), "tokenizer", None) if model is not None else None
    if tokenizer is not None:
        special_ids = set(getattr(tokenizer, "all_special_ids", []))

    # Raw-score prompts used by baseline/view_only express control as
    # "target_dim: score" rather than low/medium/high words. Match the whole
    # target phrase so a medium score like 5.00 does not accidentally select
    # every non-target dimension that is also fixed at 5.00.
    if target_score is not None:
        try:
            score_text = f"{float(target_score):.2f}"
            _add_phrase_token_matches(
                selected=selected,
                token_ids=token_ids,
                token_rows=token_rows,
                model=model,
                phrase=f"{target_dim} score of {score_text}",
                matched_keyword=(
                    f"target_score_phrase:"
                    f"{target_dim}:score_of:{score_text}"
                ),
            )
        except Exception:
            pass

    level_word = normalize_sweep_level(target_level)

    if target_dim and level_word:
        # 例如：high beautiful
        _add_phrase_token_matches(
            selected=selected,
            token_ids=token_ids,
            token_rows=token_rows,
            model=model,
            phrase=f"{level_word} {target_dim}",
            matched_keyword=(
                f"target_level_phrase:"
                f"{target_dim}:{level_word}"
            ),
        )

        # 例如：high greenery / high harmony / high charm
        for attr in DIM_ATTRS_EN.get(target_dim, []):
            _add_phrase_token_matches(
                selected=selected,
                token_ids=token_ids,
                token_rows=token_rows,
                model=model,
                phrase=f"{level_word} {attr}",
                matched_keyword=(
                    f"cue_phrase:"
                    f"{target_dim}:"
                    f"{level_word}:"
                    f"{attr}"
                ),
            )

    for kw in wanted:
        ids = _keyword_token_ids(model, kw) if model is not None else []
        if ids:
            n = len(ids)
            for start in range(0, max(0, len(token_ids) - n + 1)):
                if token_ids[start:start + n] == ids:
                    for off in range(n):
                        idx = start + off
                        if _preserve_phrase_selection(selected, idx):
                            continue
                        selected[idx] = {
                            "index": idx,
                            "token_id": int(token_ids[idx]),
                            "token": token_rows[idx]["token"],
                            "decoded": token_rows[idx]["decoded"],
                            "matched_keyword": kw,
                        }

    for row in token_rows:
        idx = int(row["index"])
        if int(row["token_id"]) in special_ids:
            continue
        cleaned_norm = cleaned.strip().lower()
        token_norm = token_raw.strip().lower()

        for kw in wanted:
            kw_norm = str(kw).strip().lower()

            exact_match = (
                cleaned_norm == kw_norm
                or token_norm == kw_norm
            )

            meaningful_subword_match = (
                len(cleaned_norm) >= 3
                and cleaned_norm in kw_norm
            )

            if exact_match or meaningful_subword_match:
                if _preserve_phrase_selection(selected, idx):
                    continue

                selected[idx] = {
                    "index": idx,
                    "token_id": int(row["token_id"]),
                    "token": row["token"],
                    "decoded": row["decoded"],
                    "matched_keyword": kw,
                }

    return [selected[k] for k in sorted(selected.keys())]


class CrossAttentionRecorder:
    """
    Added in _cross: monkey patches CrossAttention/MemoryEfficientCrossAttention
    modules and captures softmax probabilities from attn2-like calls with text
    context. The DDIM sampler in this repo calls conditional and unconditional
    branches separately; sample_images_with_cross_attention marks the branch
    before each model.apply_model call and this recorder stores conditional
    maps only.
    """
    def __init__(self, model, cfg: InferenceConfig, token_indices: List[Dict], fractions: List[float]):
        self.model = model
        self.cfg = cfg
        self.token_indices = token_indices
        self.token_index_set = {int(x["index"]) for x in token_indices}
        self.fractions = [float(x) for x in fractions]
        self.target_steps = set()
        self.target_step_fractions = {}
        self.current_phase = "off"
        self.current_step_index = None
        self.current_timestep = None
        self.enabled = False
        self.records = {}
        self.layer_counts = {}
        self.rows = []
        self.patched = []
        self.errors = []

    def configure_targets(self, total_steps: int):
        self.target_steps = set()
        self.target_step_fractions = {}
        self.total_steps = int(total_steps)
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

    def should_store(self, context):
        return (
            self.enabled
            and self.current_phase == "cond"
            and self.current_step_index in self.target_steps
            and context is not None
            and len(self.token_index_set) > 0
        )

    @staticmethod
    def layer_group(name: str) -> str:
        if "control_model" in name:
            return "control"
        if "middle_block" in name:
            return "mid"
        if "output_blocks" in name:
            return "up"
        if "input_blocks" in name:
            return "down"
        return "other"

    def store_attention(self, name: str, sim: torch.Tensor, context, query_tokens: int, token_tokens: int):
        if not self.should_store(context):
            return
        if query_tokens <= 0 or int(math.isqrt(query_tokens)) ** 2 != query_tokens:
            return
        if token_tokens <= max(self.token_index_set):
            return

        heads = getattr(self.name_to_module.get(name), "heads", 1)
        batch_heads = sim.shape[0]
        if batch_heads % heads != 0:
            return
        batch = batch_heads // heads
        attn = sim.detach().float().view(batch, heads, query_tokens, token_tokens)
        group = self.layer_group(name)
        side = int(math.isqrt(query_tokens))

        for token in self.token_indices:
            token_idx = int(token["index"])
            token_map = attn[:, :, :, token_idx].mean(dim=(0, 1)).view(side, side).detach().cpu()
            key = (group, self.current_step_index, self.current_timestep, token_idx)
            if key not in self.records:
                self.records[key] = torch.zeros_like(token_map)
                self.layer_counts[key] = 0
            self.records[key] += token_map
            self.layer_counts[key] += 1

    def patch(self):
        self.name_to_module = {}
        for name, module in self.model.named_modules():
            if not (hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v") and hasattr(module, "to_out")):
                continue
            if not hasattr(module, "heads"):
                continue
            self.name_to_module[name] = module
            original_forward = module.forward
            recorder = self

            def patched_forward(mod, x, context=None, mask=None, _name=name, _orig=original_forward):
                try:
                    h = mod.heads
                    q = mod.to_q(x)
                    context_used = default(context, x)
                    k = mod.to_k(context_used)
                    v = mod.to_v(context_used)

                    if hasattr(mod, "dim_head"):
                        dim_head = mod.dim_head
                        b = q.shape[0]
                        q, k, v = map(
                            lambda t: t.unsqueeze(3)
                            .reshape(b, t.shape[1], h, dim_head)
                            .permute(0, 2, 1, 3)
                            .reshape(b * h, t.shape[1], dim_head)
                            .contiguous(),
                            (q, k, v),
                        )
                        scale = dim_head ** -0.5
                    else:
                        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
                        scale = mod.scale

                    sim = torch.einsum("b i d, b j d -> b i j", q.float(), k.float()) * scale
                    if exists(mask):
                        mask2 = einops.rearrange(mask, "b ... -> b (...)")
                        max_neg = -torch.finfo(sim.dtype).max
                        mask2 = einops.repeat(mask2, "b j -> (b h) () j", h=h)
                        sim.masked_fill_(~mask2, max_neg)
                    sim = sim.softmax(dim=-1)
                    recorder.store_attention(_name, sim, context, x.shape[1], context_used.shape[1])
                    out = torch.einsum("b i j, b j d -> b i d", sim.to(v.dtype), v)
                    out = einops.rearrange(out, "(b h) n d -> b n (h d)", h=h)
                    return mod.to_out(out)
                except Exception as e:
                    # Keep inference alive if a backend-specific attention shape
                    # differs from the vanilla path. The module then runs with
                    # its original forward, but this one attention call is not
                    # recorded.
                    recorder.errors.append(f"{_name}: {e}")
                    return _orig(x, context=context, mask=mask)

            module.forward = types.MethodType(patched_forward, module)
            self.patched.append((module, original_forward))

    def restore(self):
        for module, original_forward in self.patched:
            module.forward = original_forward
        self.patched = []

    def save(self, out_dir: Path, token_info: Dict, sample_meta: Dict, generated_path: Optional[Path]) -> Tuple[List[Dict], Optional[str]]:
        ensure_dir(out_dir)
        token_payload = dict(token_info)
        token_payload["selected_tokens"] = self.token_indices
        token_payload["selection_warning"] = "" if self.token_indices else "No requested level/dimension/cue token was found."
        token_payload["attention_capture_warnings"] = self.errors[:50]
        (out_dir / "tokens.json").write_text(json.dumps(token_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        rows = []
        token_by_idx = {int(x["index"]): x for x in self.token_indices}
        phrase_accumulators: Dict[Tuple[str, int, int, str], Dict] = {}
        for (group, step_idx, timestep, token_idx), acc in sorted(self.records.items(), key=lambda x: (x[0][1], x[0][0], x[0][3])):
            count = max(int(self.layer_counts[(group, step_idx, timestep, token_idx)]), 1)
            attn_map = (acc / count).numpy().astype(np.float32)
            token = token_by_idx.get(token_idx, {"token": str(token_idx), "decoded": str(token_idx), "matched_keyword": ""})
            matched_keyword = str(token.get("matched_keyword", ""))
            if _is_phrase_match(matched_keyword):
                phrase_key = (group, int(step_idx), int(timestep), matched_keyword)
                if phrase_key not in phrase_accumulators:
                    phrase_accumulators[phrase_key] = {
                        "maps": [],
                        "token_indices": [],
                        "token_texts": [],
                        "layer_counts": [],
                    }
                phrase_accumulators[phrase_key]["maps"].append(attn_map)
                phrase_accumulators[phrase_key]["token_indices"].append(int(token_idx))
                phrase_accumulators[phrase_key]["token_texts"].append(token.get("decoded", token.get("token", "")))
                phrase_accumulators[phrase_key]["layer_counts"].append(count)
            token_name = sanitize_token_for_filename(token.get("decoded") or token.get("token"))
            frac = float(self.target_step_fractions.get(step_idx, -1.0))
            frac_tag = f"{frac:.2f}".replace(".", "p")
            base_name = f"attn_{group}_step_{int(step_idx):03d}_frac_{frac_tag}_token_{token_idx:02d}_{token_name}"

            npy_path = out_dir / f"{base_name}.npy"
            png_path = out_dir / f"{base_name}.png"
            if CROSS_ATTENTION_SAVE_NUMPY:
                np.save(npy_path, resize_float_map(attn_map, self.cfg.image_resolution).astype(np.float32))
            if CROSS_ATTENTION_SAVE_PNG:
                save_attention_png(
                    attn_map,
                    png_path,
                    self.cfg.image_resolution,
                    overlay_path=generated_path if CROSS_ATTENTION_SAVE_OVERLAY else None,
                )

            row = {
                **sample_meta,
                "ddim_step_index": int(step_idx),
                "original_timestep_value": int(timestep),
                "timestep_fraction": frac,
                "total_ddim_steps": int(getattr(self, "total_steps", self.cfg.ddim_steps)),
                "attention_layer_name": group,
                "attention_layer_group": group,
                "attention_map_shape": f"{attn_map.shape[0]}x{attn_map.shape[1]}",
                "attention_layers_aggregated": count,
                "attention_unit": "token",
                "token_index": int(token_idx),
                "token_id": int(token.get("token_id", -1)),
                "token_text": token.get("decoded", token.get("token", "")),
                "matched_keyword": matched_keyword,
                "attention_npy_path": str(npy_path) if CROSS_ATTENTION_SAVE_NUMPY else "",
                "attention_png_path": str(png_path) if CROSS_ATTENTION_SAVE_PNG else "",
                "attention_branch": "conditional",
            }
            rows.append(row)

        for (group, step_idx, timestep, matched_keyword), payload in sorted(
            phrase_accumulators.items(), key=lambda x: (x[0][1], x[0][0], x[0][3])
        ):
            if not payload["maps"]:
                continue
            phrase_map = np.mean(np.stack(payload["maps"], axis=0), axis=0).astype(np.float32)
            frac = float(self.target_step_fractions.get(step_idx, -1.0))
            frac_tag = f"{frac:.2f}".replace(".", "p")
            phrase_name = sanitize_token_for_filename(matched_keyword)
            base_name = f"attn_{group}_step_{int(step_idx):03d}_frac_{frac_tag}_phrase_{phrase_name}"
            npy_path = out_dir / f"{base_name}.npy"
            png_path = out_dir / f"{base_name}.png"
            if CROSS_ATTENTION_SAVE_NUMPY:
                np.save(npy_path, resize_float_map(phrase_map, self.cfg.image_resolution).astype(np.float32))
            if CROSS_ATTENTION_SAVE_PNG:
                save_attention_png(
                    phrase_map,
                    png_path,
                    self.cfg.image_resolution,
                    overlay_path=generated_path if CROSS_ATTENTION_SAVE_OVERLAY else None,
                )

            rows.append({
                **sample_meta,
                "ddim_step_index": int(step_idx),
                "original_timestep_value": int(timestep),
                "timestep_fraction": frac,
                "total_ddim_steps": int(getattr(self, "total_steps", self.cfg.ddim_steps)),
                "attention_layer_name": group,
                "attention_layer_group": group,
                "attention_map_shape": f"{phrase_map.shape[0]}x{phrase_map.shape[1]}",
                "attention_layers_aggregated": int(max(payload["layer_counts"]) if payload["layer_counts"] else 1),
                "attention_unit": "phrase_average",
                "token_index": -1,
                "token_id": -1,
                "token_text": " ".join(str(x) for x in payload["token_texts"]),
                "token_indices": ",".join(str(x) for x in payload["token_indices"]),
                "matched_keyword": matched_keyword,
                "attention_npy_path": str(npy_path) if CROSS_ATTENTION_SAVE_NUMPY else "",
                "attention_png_path": str(png_path) if CROSS_ATTENTION_SAVE_PNG else "",
                "attention_branch": "conditional",
            })

        meta_path = out_dir / "cross_attention_metadata.csv"
        pd.DataFrame(rows).to_csv(meta_path, index=False, encoding="utf-8-sig")
        return rows, str(meta_path)


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
            cond_txt = torch.cat(cond_arg["c_crossattn"], 1)
            raw_control = torch.cat(cond_arg["c_concat"], 1)
            control_hint = self.model.prepare_control_hint(raw_control, cond_arg)
            control = self.model.control_model(x=x_noisy, hint=control_hint, timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.model.control_scales)]

            max_blocks = CONTROL_ANALYSIS_MAX_BLOCKS
            for block_idx, c in enumerate(control):
                if max_blocks is not None and block_idx >= int(max_blocks):
                    break
                self.records.append({
                    "block_index": int(block_idx),
                    "block_name": f"control_block_{block_idx:02d}",
                    "ddim_step_index": int(self.current_step_index),
                    "original_timestep_value": int(self.current_timestep),
                    "timestep_fraction": float(self.target_step_fractions.get(self.current_step_index, -1.0)),
                    "total_ddim_steps": int(self.total_steps),
                    "tensor_shape": "x".join(str(x) for x in c.shape),
                    "heatmap": heatmap_from_feature_tensor(c, self.cfg.image_resolution),
                    **feature_summary_stats(c),
                })
        except Exception as e:
            self.errors.append(f"control residual capture failed at step {key}: {e}")

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


# ============================================================
# 7. Sampling
# ============================================================

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


@torch.no_grad()
def sample_images(model, sampler, control: torch.Tensor, prompt: str, cfg: InferenceConfig) -> np.ndarray:
    bsz = cfg.num_samples

    if control.shape[0] == 1 and bsz > 1:
        control_b = control.repeat(bsz, 1, 1, 1)
    else:
        control_b = control

    cond = {
        "c_concat": [control_b],
        "c_crossattn": [model.get_learned_conditioning([prompt] * bsz)],
    }
    un_cond = {
        "c_concat": [control_b],
        "c_crossattn": [model.get_learned_conditioning([cfg.n_prompt] * bsz)],
    }

    shape = (4, cfg.image_resolution // 8, cfg.image_resolution // 8)
    model.control_scales = [cfg.strength] * 13

    samples, _ = sampler.sample(
        cfg.ddim_steps,
        bsz,
        shape,
        cond,
        eta=cfg.eta,
        unconditional_guidance_scale=cfg.guidance_scale,
        unconditional_conditioning=un_cond,
    )

    x = model.decode_first_stage(samples)
    x = einops.rearrange(x, "b c h w -> b h w c")
    x = (x * 127.5 + 127.5).detach().cpu().numpy()
    return np.clip(x, 0, 255).astype(np.uint8)


@torch.no_grad()
def sample_images_with_cross_attention(
    model,
    sampler,
    control: torch.Tensor,
    prompt: str,
    cfg: InferenceConfig,
    attention_out_dir: Optional[Path],
    sample_meta: Dict,
    generated_path_for_overlay: Optional[Path] = None,
    control_out_dir: Optional[Path] = None,
    raw_hint_path: Optional[Path] = None,
    view_hint_path: Optional[Path] = None,
    capture_attention: bool = True,
    capture_control: bool = False,
) -> Tuple[np.ndarray, List[Dict], Optional[str], List[Dict], Optional[str], List[str]]:
    token_info = get_clip_token_records(model, prompt)
    selected_tokens = select_token_indices(
        token_info=token_info,
        prompt=prompt,
        target_dim=sample_meta.get("target_dim", ""),
        target_level=sample_meta.get("target_level", ""),
        keywords=CROSS_ATTENTION_KEYWORDS,
        model=model,
        target_score=sample_meta.get("target_score_for_raw_prompt", None),
    )

    bsz = cfg.num_samples
    control_b = control.repeat(bsz, 1, 1, 1) if control.shape[0] == 1 and bsz > 1 else control
    cond = {
        "c_concat": [control_b],
        "c_crossattn": [model.get_learned_conditioning([prompt] * bsz)],
    }
    un_cond = {
        "c_concat": [control_b],
        "c_crossattn": [model.get_learned_conditioning([cfg.n_prompt] * bsz)],
    }

    shape = (4, cfg.image_resolution // 8, cfg.image_resolution // 8)
    model.control_scales = [cfg.strength] * 13

    attn_recorder = CrossAttentionRecorder(model, cfg, selected_tokens, ANALYSIS_TIMESTEP_FRACTIONS) if capture_attention else None
    if attn_recorder is not None:
        attn_recorder.configure_targets(cfg.ddim_steps)
    control_recorder = ControlAnalysisRecorder(model, cfg, ANALYSIS_TIMESTEP_FRACTIONS) if capture_control else None
    if control_recorder is not None:
        control_recorder.configure_targets(cfg.ddim_steps)
    original_apply_model = model.apply_model

    def wrapped_apply_model(x_noisy, t, cond_arg, *args, **kwargs):
        if attn_recorder is not None:
            attn_recorder.set_context(sampler, cond, cond_arg, t)
        if control_recorder is not None:
            control_recorder.set_context(sampler, cond, cond_arg, t)
            old_attn_enabled = attn_recorder.enabled if attn_recorder is not None else None
            if attn_recorder is not None:
                attn_recorder.enabled = False
            control_recorder.capture(x_noisy, t, cond_arg)
            if attn_recorder is not None:
                attn_recorder.enabled = old_attn_enabled
        try:
            return original_apply_model(x_noisy, t, cond_arg, *args, **kwargs)
        finally:
            if attn_recorder is not None:
                attn_recorder.clear_context()
            if control_recorder is not None:
                control_recorder.clear_context()

    try:
        if attn_recorder is not None:
            attn_recorder.enabled = True
            attn_recorder.patch()
        model.apply_model = wrapped_apply_model
        samples, _ = sampler.sample(
            cfg.ddim_steps,
            bsz,
            shape,
            cond,
            eta=cfg.eta,
            unconditional_guidance_scale=cfg.guidance_scale,
            unconditional_conditioning=un_cond,
        )
    finally:
        model.apply_model = original_apply_model
        if attn_recorder is not None:
            attn_recorder.restore()

    x = model.decode_first_stage(samples)
    x = einops.rearrange(x, "b c h w -> b h w c")
    x = (x * 127.5 + 127.5).detach().cpu().numpy()
    imgs = np.clip(x, 0, 255).astype(np.uint8)
    if generated_path_for_overlay is not None and imgs.shape[0] > 0:
        gp = Path(generated_path_for_overlay)
        if not gp.exists():
            ensure_dir(gp.parent)
            Image.fromarray(imgs[0]).save(gp, quality=95)
    attn_rows, attn_metadata_path = [], None
    if attn_recorder is not None and attention_out_dir is not None:
        attn_rows, attn_metadata_path = attn_recorder.save(attention_out_dir, token_info, sample_meta, generated_path_for_overlay)
    control_rows, control_metadata_path, control_errors = [], None, []
    if control_recorder is not None and control_out_dir is not None:
        control_rows, control_metadata_path, control_errors = control_recorder.save(
            control_out_dir,
            sample_meta,
            raw_hint_path=raw_hint_path,
            view_hint_path=view_hint_path,
            generated_path=generated_path_for_overlay,
        )
    return imgs, attn_rows, attn_metadata_path, control_rows, control_metadata_path, control_errors


def save_generated_images(imgs: np.ndarray, out_path_no_sample: Path, overwrite: bool = False) -> List[Path]:
    ensure_dir(out_path_no_sample.parent)
    saved = []

    if imgs.shape[0] == 1:
        out_path = out_path_no_sample
        if out_path.exists() and not overwrite:
            return [out_path]
        Image.fromarray(imgs[0]).save(out_path, quality=95)
        return [out_path]

    stem = out_path_no_sample.stem
    suffix = out_path_no_sample.suffix
    for i in range(imgs.shape[0]):
        out_path = out_path_no_sample.with_name(f"{stem}_s{i:02d}{suffix}")
        if out_path.exists() and not overwrite:
            saved.append(out_path)
            continue
        Image.fromarray(imgs[i]).save(out_path, quality=95)
        saved.append(out_path)
    return saved


# ============================================================
# 7. Reconstruction
# ============================================================

def run_reconstruction_for_experiment(cfg: InferenceConfig, df: pd.DataFrame, exp_name: str, ckpt_selector: str, device: torch.device) -> pd.DataFrame:
    model, sampler, ckpt_path = load_model_and_sampler(cfg, exp_name, ckpt_selector, device)
    set_global_seed(cfg.seed + stable_int_hash(exp_name))

    ckpt_tag = ckpt_path.stem
   
    shared_root = cfg.output_root / "reconstruction"
    real_dir = ensure_dir(shared_root / "real")
    hint_dir = ensure_dir(shared_root / "hint")

    gen_base = ensure_dir(shared_root / "generated" / exp_name)
    gen_dir = ensure_dir(gen_base / ckpt_tag)

    prompt_base = ensure_dir(shared_root / "prompts" / exp_name)
    prompt_dir = prompt_base   # 已经存在，无需再 ensure_dir

    metadata_rows = []
    rows = select_reconstruction_rows(df, cfg)
    print(f"[RECON] {exp_name}: {len(rows)} samples")

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"reconstruction/{exp_name}"):
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
        if img_path is None or hint_path is None:
            continue

        prompt = build_prompt_for_experiment(row, cfg, exp_name)
        out_img = gen_dir / f"{image_id}.png"

        if out_img.exists() and not cfg.overwrite:
            generated_paths = [out_img]
        else:
            try:
                control = process_hint_image(hint_path, cfg, device)
                sample_seed = cfg.seed + stable_int_hash(f"recon-{image_id}")
                set_global_seed(sample_seed)
                imgs = sample_images(model, sampler, control, prompt, cfg)
                generated_paths = save_generated_images(imgs, out_img, overwrite=cfg.overwrite)
            except Exception as e:
                print(f"[RECON-SKIP] {exp_name}/{image_id}: {e}")
                continue

        prompt_path = prompt_dir / f"{image_id}.txt"
        if cfg.overwrite or not prompt_path.exists():
            prompt_path.write_text(prompt, encoding="utf-8")

        if cfg.save_real_and_hint:
            safe_copy_or_save_image(img_path, real_dir / f"{image_id}.png", cfg.image_resolution)
            safe_copy_or_save_image(hint_path, hint_dir / f"{image_id}.png", cfg.image_resolution)

        for gp in generated_paths:
            meta = {
                "mode": "reconstruction",
                "experiment": exp_name,
                "ckpt_selector": ckpt_selector,
                "ckpt_path": str(ckpt_path),
                "image_id": image_id,
                "image_name_raw": image_name,
                "generated_path": str(gp),
                "real_path": str(real_dir / f"{image_id}.png"),
                "hint_path": str(hint_dir / f"{image_id}.png"),
                "prompt_path": str(prompt_path),
                "prompt": prompt,
            }
            for dim in cfg.dimensions:
                meta[f"{dim}_score"] = get_dim_score(row, dim)
                meta[f"{dim}_level"] = get_dim_level(row, dim)
            metadata_rows.append(meta)

    meta_df = pd.DataFrame(metadata_rows)
    meta_path = prompt_dir / "metadata_reconstruction.csv"
    meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"[RECON] metadata saved: {meta_path}")

    del model, sampler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return meta_df


# ============================================================
# 8. Sweep with fixed TARGET_IMAGE_IDS
# ============================================================

def run_sweep_for_experiment(cfg: InferenceConfig, df: pd.DataFrame, exp_name: str, ckpt_selector: str, device: torch.device) -> pd.DataFrame:
    model, sampler, ckpt_path = load_model_and_sampler(cfg, exp_name, ckpt_selector, device)
    set_global_seed(cfg.seed + stable_int_hash(exp_name))

    ckpt_tag = ckpt_path.stem
    out_base = cfg.output_root / "sweep" / exp_name / ckpt_tag
    prompt_dir = ensure_dir(out_base / "prompts")
    sweep_shared_root = cfg.output_root / "sweep"
    sweep_real_dir = ensure_dir(sweep_shared_root / "real")
    sweep_bev_dir = ensure_dir(sweep_shared_root / "BEV")

    metadata_rows = []
    rows = select_sweep_rows(df, cfg)
    print(
        f"[SWEEP] {exp_name}: fixed {len(rows)} images × "
        f"{len(cfg.dimensions)} dims × {len(cfg.sweep_levels)} levels"
    )

    cross_attention_rows = []
    control_analysis_rows = []

    for image_index, (_, row) in enumerate(tqdm(rows.iterrows(), total=len(rows), desc=f"sweep/{exp_name}")):

        exp_use_view_transform = bool(EXPERIMENTS[exp_name]["use_view_transform"])

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
            print(f"[SWEEP-SKIP] missing image: {image_id}")
            continue
        if hint_path is None:
            print(f"[SWEEP-SKIP] missing hint: {image_id}")
            continue

        shared_real_path = sweep_real_dir / f"{image_id}.png"
        shared_bev_path = sweep_bev_dir / f"{image_id}.png"
        if cfg.overwrite or not shared_real_path.exists():
            safe_copy_or_save_image(img_path, shared_real_path, cfg.image_resolution)
        if cfg.overwrite or not shared_bev_path.exists():
            safe_copy_or_save_image(hint_path, shared_bev_path, cfg.image_resolution)

        try:
            control = process_hint_image(hint_path, cfg, device)
        except Exception as e:
            print(f"[SWEEP-SKIP] {exp_name}/{image_id}: {e}")
            continue

        shared_view_hint_path = ""
        shared_view_hint_error = ""
        if SAVE_VIEW_HINT:
            shared_view_hint_path, shared_view_hint_error = get_or_save_shared_view_hint(
                model=model,
                control=control,
                out_base=sweep_shared_root,
                image_id=image_id,
                use_view_transform=exp_use_view_transform,
                image_resolution=cfg.image_resolution,
                overwrite=cfg.overwrite,
            )
            if shared_view_hint_error:
                print(f"[VIEW-HINT-WARN] {exp_name}/{image_id}: {shared_view_hint_error}")

        for target_dim in cfg.dimensions:
            for level_raw in cfg.sweep_levels:
                level_word = normalize_sweep_level(level_raw)

                override_levels, override_scores = build_sweep_overrides(
                    row=row,
                    cfg=cfg,
                    target_dim=target_dim,
                    target_level=level_word,
                    exp_name=exp_name,
                )
                if exp_name == "full":
                    prompt = build_level_cue_prompt_for_target_dim(
                        row=row,
                        cfg=cfg,
                        target_dim=target_dim,
                        override_levels=override_levels,
                        include_other_dim_targets=True,
                        include_other_dim_cues=False,
                    )
                else:
                    prompt = build_prompt_for_experiment(
                        row=row,
                        cfg=cfg,
                        exp_name=exp_name,
                        override_levels=override_levels,
                        override_scores=override_scores,
                    )

                level_tag = level_word.replace(" ", "_")
                sub_dir = ensure_dir(out_base / target_dim / image_id)
                out_img = sub_dir / f"{image_id}_{target_dim}_{level_tag}.png"

                view_hint_path = shared_view_hint_path
                view_hint_error = shared_view_hint_error

                capture_attention = should_capture_cross_attention(exp_name, target_dim, level_word, image_index)
                cross_attention_dir = (
                    out_base / "cross_attention" / target_dim / image_id / level_tag
                    if capture_attention else None
                )
                capture_control = should_capture_control_analysis(exp_name, target_dim, level_word, image_index)
                control_analysis_dir = (
                    out_base / "control_analysis" / target_dim / image_id / level_tag
                    if capture_control else None
                )
                cross_attention_metadata_path = ""
                cross_attention_error = ""
                control_analysis_metadata_path = ""
                control_analysis_error = ""

                if out_img.exists() and not cfg.overwrite:
                    generated_paths = [out_img]
                    if capture_attention:
                        cross_attention_error = "Generated image already exists and OVERWRITE=False; attention was not re-sampled."
                    if capture_control:
                        control_analysis_error = "Generated image already exists and OVERWRITE=False; control analysis was not re-sampled."
                else:
                    try:
                        if cfg.same_latent_per_sweep:
                            sample_seed = cfg.seed + stable_int_hash(
                                f"sweep-{image_id}-{target_dim}"
                            )
                        else:
                            sample_seed = cfg.seed + stable_int_hash(
                                f"sweep-{exp_name}-{image_id}-{target_dim}-{level_word}"
                            )
                        set_global_seed(sample_seed)

                        sample_meta_for_attention = {
                            "experiment": exp_name,
                            "ckpt_tag": ckpt_tag,
                            "image_id": image_id,
                            "target_dim": target_dim,
                            "target_level": level_word,
                            "target_score_for_raw_prompt": override_scores.get(target_dim, sweep_level_to_score(level_word)),
                            "generated_path": str(out_img),
                            "real_path": str(shared_real_path),
                            "bev_hint_path": str(shared_bev_path),
                            "view_hint_path": view_hint_path,
                            "raw_hint_path": str(hint_path),
                            "prompt": prompt,
                        }
                        if capture_attention or capture_control:
                            try:
                                (
                                    imgs,
                                    attn_rows,
                                    cross_attention_metadata_path,
                                    control_rows,
                                    control_analysis_metadata_path,
                                    control_errors,
                                ) = sample_images_with_cross_attention(
                                    model=model,
                                    sampler=sampler,
                                    control=control,
                                    prompt=prompt,
                                    cfg=cfg,
                                    attention_out_dir=cross_attention_dir,
                                    sample_meta=sample_meta_for_attention,
                                    generated_path_for_overlay=out_img,
                                    control_out_dir=control_analysis_dir,
                                    raw_hint_path=hint_path,
                                    view_hint_path=Path(view_hint_path) if view_hint_path else None,
                                    capture_attention=capture_attention,
                                    capture_control=capture_control,
                                )
                                cross_attention_rows.extend(attn_rows)
                                control_analysis_rows.extend(control_rows)
                                if control_errors:
                                    control_analysis_error = " | ".join(control_errors)
                            except Exception as e:
                                if capture_attention:
                                    cross_attention_error = str(e)
                                if capture_control:
                                    control_analysis_error = str(e)
                                print(f"[ANALYSIS-WARN] {exp_name}/{image_id}/{target_dim}/{level_word}: {e}")
                                set_global_seed(sample_seed)
                                imgs = sample_images(model, sampler, control, prompt, cfg)
                        else:
                            imgs = sample_images(model, sampler, control, prompt, cfg)
                        generated_paths = save_generated_images(imgs, out_img, overwrite=cfg.overwrite)
                    except Exception as e:
                        print(f"[SWEEP-SKIP] {exp_name}/{image_id}/{target_dim}/{level_word}: {e}")
                        if capture_attention:
                            cross_attention_error = str(e)
                        continue

                prompt_path = prompt_dir / f"{image_id}_{target_dim}_{level_tag}.txt"
                if cfg.overwrite or not prompt_path.exists():
                    prompt_path.write_text(prompt, encoding="utf-8")

                for gp in generated_paths:
                    meta = {
                        "mode": "sweep",
                        "experiment": exp_name,
                        "ckpt_path": str(ckpt_path),
                        "image_id": image_id,
                        "image_name_raw": image_name,
                        "target_dim": target_dim,
                        "target_level": level_word,
                        "target_score_for_raw_prompt": sweep_level_to_score(level_word),
                        "sweep_context": cfg.sweep_context,
                        "generated_path": str(gp),
                        "real_path": str(shared_real_path),
                        "bev_hint_path": str(shared_bev_path),
                        "source_hint_path": str(hint_path),
                        "raw_hint_path": str(hint_path),
                        "view_hint_path": view_hint_path,
                        "view_hint_error": view_hint_error,
                        "use_view_transform": exp_use_view_transform,
                        "control_is_projected": False,
                        "cross_attention_dir": str(cross_attention_dir) if cross_attention_dir is not None else "",
                        "cross_attention_metadata_path": cross_attention_metadata_path,
                        "cross_attention_error": cross_attention_error,
                        "control_analysis_dir": str(control_analysis_dir) if control_analysis_dir is not None else "",
                        "control_analysis_metadata_path": control_analysis_metadata_path,
                        "control_analysis_error": control_analysis_error,
                        "prompt_path": str(prompt_path),
                        "prompt": prompt,
                    }
                    for dim in cfg.dimensions:
                        meta[f"original_{dim}_score"] = get_dim_score(row, dim)
                        meta[f"original_{dim}_level"] = get_dim_level(row, dim)
                        if dim in override_levels:
                            meta[f"used_{dim}_level"] = override_levels[dim]
                        if dim in override_scores:
                            meta[f"used_{dim}_score"] = override_scores[dim]
                    metadata_rows.append(meta)

    meta_df = pd.DataFrame(metadata_rows)
    meta_path = out_base / "metadata_sweep.csv"
    meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"[SWEEP] metadata saved: {meta_path}")

    if cross_attention_rows:
        ca_meta_path = out_base / "cross_attention_metadata.csv"
        pd.DataFrame(cross_attention_rows).to_csv(ca_meta_path, index=False, encoding="utf-8-sig")
        print(f"[CROSS-ATTN] combined metadata saved: {ca_meta_path}")

    if control_analysis_rows:
        ctrl_meta_path = out_base / "control_analysis_metadata.csv"
        pd.DataFrame(control_analysis_rows).to_csv(ctrl_meta_path, index=False, encoding="utf-8-sig")
        print(f"[CONTROL] combined metadata saved: {ctrl_meta_path}")

    del model, sampler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return meta_df


# ============================================================
# 9. Main
# ============================================================

def print_config(cfg: InferenceConfig, device: torch.device) -> None:
    print("\n========== Inference Config ==========")
    print(f"RUN_MODE          : {cfg.run_mode}")
    print(f"EXPERIMENT_ORDER  : {cfg.experiment_order}")
    print(f"DIMENSIONS        : {cfg.dimensions}")
    print(f"SPLIT             : {cfg.split}")
    print(f"STANDARDIZED_CSV  : {cfg.standardized_csv}")
    print(f"IMAGE_DIR         : {cfg.image_dir}")
    print(f"LAYOUT_DIR        : {cfg.layout_dir}")
    print(f"MODEL_CONFIG_PATH : {cfg.model_config_path}")
    print(f"CKPT_ROOT         : {cfg.ckpt_root}")
    print(f"CKPT_NAMES        : {cfg.ckpt_names}")
    print(f"OUTPUT_ROOT       : {cfg.output_root}")
    print(f"DEVICE            : {device}")
    print(f"MAX_RECON         : {cfg.max_recon}")
    print(f"TARGET_IMAGE_IDS  : {len(cfg.target_image_ids)} fixed images")
    print(f"SWEEP_LEVELS      : {cfg.sweep_levels}")
    print(f"SWEEP_CONTEXT     : {cfg.sweep_context}")
    print(f"DDIM_STEPS        : {cfg.ddim_steps}")
    print(f"GUIDANCE_SCALE    : {cfg.guidance_scale}")
    print(f"STRENGTH          : {cfg.strength}")
    print(f"SEED              : {cfg.seed}")
    print("======================================\n")


def main() -> None:
    cfg = build_config_from_user_settings()

    if cfg.run_mode not in ["reconstruction", "sweep", "both"]:
        raise ValueError("RUN_MODE must be 'reconstruction', 'sweep', or 'both'.")

    for exp_name in cfg.experiment_order:
        if exp_name not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {exp_name}. Available: {list(EXPERIMENTS.keys())}")

    if cfg.sweep_context not in ["medium", "row"]:
        raise ValueError("SWEEP_CONTEXT must be 'medium' or 'row'.")

    device = get_device(cfg.device)
    set_global_seed(cfg.seed)
    ensure_dir(cfg.output_root)
    print_config(cfg, device)

    df, image_col = load_label_table(cfg)
    print(f"[DATA] image column: {image_col}")

    all_meta = []

    for exp_name in cfg.experiment_order:
        print(f"\n========== START EXPERIMENT: {exp_name} ==========")
        print(f"[EXP CFG] {EXPERIMENTS[exp_name]}")

        for ckpt_selector in cfg.ckpt_names:
            print(f"---- checkpoint selector: {ckpt_selector} ----")

            if cfg.run_mode in ["reconstruction", "both"]:
                meta_recon = run_reconstruction_for_experiment(cfg, df, exp_name, ckpt_selector, device)
                all_meta.append(meta_recon)

            if cfg.run_mode in ["sweep", "both"]:
                meta_sweep = run_sweep_for_experiment(cfg, df, exp_name, ckpt_selector, device)
                all_meta.append(meta_sweep)

        print(f"========== END EXPERIMENT: {exp_name} ==========\n")

    if all_meta:
        combined = pd.concat(all_meta, ignore_index=True)
        combined_path = cfg.output_root / f"metadata_all_{cfg.run_mode}.csv"
        combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"[DONE] combined metadata saved: {combined_path}")

    print("\n全部推理完成。")


if __name__ == "__main__":
    # Cross-analysis output notes:
    # - ANALYSIS_TIMESTEP_FRACTIONS is shared by prompt and hint analyses:
    #   early ~= structure formation, middle ~= structure/semantic coupling,
    #   late ~= visual detail refinement.
    # - sweep/real/<image_id>.png and sweep/BEV/<image_id>.png collect the GT
    #   street-view image and raw BEV hint once for the fixed sweep images.
    # - sweep/view_hint/<image_id>_view_hint.png stores one reusable projected
    #   view per image because it depends on hint geometry, not experiment,
    #   checkpoint, dim, or level.
    #   It contains the model projector's
    #   street-view RGB hint: road is orange, building is blue, and white means
    #   no projected geometry.
    # - cross_attention/<dim>/<image_id>/<level>/tokens.json records CLIP token
    #   ids/text and selected level/dimension/cue tokens.
    # - cross_attention/*/attn_<group>_step_*_frac_* files store conditional-only,
    #   head-averaged token maps resized to IMAGE_RESOLUTION for alignment with
    #   generated images and view_hint PNGs.
    # - control_analysis/<dim>/<image_id>/<level>/ stores ControlNet residual
    #   and middle-delta maps for BEV/view-hint control mechanism analysis.
    # - Compare cross-attention and residual maps only after 0-1 normalization
    #   and at matched timestep_fraction / DDIM step; raw values have different
    #   units and should not be interpreted as directly comparable.
    main()

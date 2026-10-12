# -*- coding: utf-8 -*-


from __future__ import annotations

import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import einops
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm

from cldm.model import create_model, load_state_dict as cldm_load_state_dict
from cldm.ddim_hacked import DDIMSampler


# ============================================================
# 0. USER SETTINGS：只改这里即可
# ============================================================

# ---------- What to run ----------
# 可选："reconstruction" / "sweep" / "both"
RUN_MODE = "both"

# 可选模型组："baseline", "view_only", "view_level_only", "full"
EXPERIMENT_ORDER = [
"baseline", "view_only",
    "view_level_only",
    "full",
]

# 六个 perception 维度。顺序会影响 prompt 中维度排列。
DIMENSIONS = ["beautiful", "safety", "wealthy", "boring", "depressing", "lively"]

# 推理使用哪个 split。
# 如果 CSV 中有 split 列，则只取对应 split；如果没有 split 列，会自动使用全部样本。
# 可选："train" / "val" / "test" / "all"
SPLIT = "val"


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
    "epoch9-step67920.ckpt",
]

OUTPUT_ROOT = BASE_DATA_DIR / "inference_outputs_multidim_ablation"


# ---------- Reconstruction settings ----------
# reconstruction 是从 SPLIT 中抽样做重建；设为 None 或 <=0 表示使用全部可用样本。
MAX_RECON = 1000


# ---------- Sweep settings ----------
# 这里是 sweep 固定使用的图片。
# 也就是说：sweep 不再随机抽 max_sweep_images，而是只对下面这些 image_id 做固定对比。
TARGET_IMAGE_IDS = [
    "50e5faebd7c3df413b000743",
    "50f5e89afdc9f065f0007409",
    "50f5e562fdc9f065f00067db",
    "50f5ed9bfdc9f065f0008e46",
    "50f5eeb1fdc9f065f000901a",
    "50f439f1fdc9f065f0002d7b",
    "50f607e1beb2fed6f8000388",
    "513e6146fdc9f0358700b50d",
    "51421a69fdc9f04926008339",
    "513d3836fdc9f03587002c41",
    "50f5e640fdc9f065f0006bb6",
    "50e5f7d6d7c3df413b000583",
    "50e5f7d6d7c3df413b000582",
    "513d57f6fdc9f03587003262",
    "50f5ee99fdc9f065f0008f32",
    "50f5ecf0fdc9f065f0008b2b",
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
APPEND_QUALITY_PROMPT = True
STATIC_TAIL = "Follow only static urban features (no people, no vehicles)."

# 是否保存 real / hint 到 reconstruction 输出文件夹
SAVE_REAL_AND_HINT = True

# 如果 False，遇到已经生成的图片会跳过。
OVERWRITE = False


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
DIM_ATTRS_EN: Dict[str, List[str]] = {
    "safety": [
        "road-interface continuity",
        "edge clarity",
        "greenery buffers",
        "vegetation-led enclosure",
        "clear road alignment",
    ],
    "beautiful": [
        "color saturation",
        "visible greenery",
        "soft lighting",
        "textural richness",
        "clutter reduction",
    ],
    "lively": [
        "edge/detail density",
        "street-wall enclosure",
        "mixed façades & greenery",
        "textural richness",
        "reduced open sky",
    ],
    "wealthy": [
        "greenery abundance",
        "road regularity",
        "façade neatness",
        "color depth",
        "wayfinding clarity",
    ],
    "depressing": [
        "visual desaturation",
        "flat glaring light",
        "empty sky openness",
        "greenery sparsity",
        "road-surface dominance",
    ],
    "boring": [
        "bright-but-desaturated look",
        "sky/open area",
        "greenery sparsity",
        "detail lack",
        "expansive road surface",
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


# ============================================================
# 3. Data loading
# ============================================================

def detect_image_col(df: pd.DataFrame) -> str:
    candidates = ["image_name", "image_id", "img_id", "filename", "file_name", "name"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find image id column. Tried {candidates}. Existing columns: {list(df.columns)}")


def detect_split_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["split", "dataset_split", "phase", "train_val_test"]:
        if c in df.columns:
            return c
    return None


def load_label_table(cfg: InferenceConfig) -> Tuple[pd.DataFrame, str]:
    if not cfg.standardized_csv.exists():
        raise FileNotFoundError(f"standardized_csv not found: {cfg.standardized_csv}")

    df = pd.read_csv(cfg.standardized_csv)
    image_col = detect_image_col(df)

    df["_image_name_raw"] = df[image_col].astype(str)
    df["_image_id"] = df[image_col].astype(str).map(stem_no_ext)

    split_col = detect_split_col(df)
    if cfg.split.lower() != "all":
        if split_col is not None:
            before = len(df)
            df = df[df[split_col].astype(str).str.lower() == cfg.split.lower()].copy()
            print(f"[DATA] split={cfg.split}, {before} -> {len(df)} rows by column '{split_col}'")
        else:
            print(f"[WARN] split={cfg.split} requested, but no split column was found. Use all rows.")

    valid_rows = []
    missing_image = 0
    missing_hint = 0

    for _, row in df.iterrows():
        image_name = str(row["_image_name_raw"])
        image_id = str(row["_image_id"])
        img_path = find_file(cfg.image_dir, image_name) or find_file(cfg.image_dir, image_id)
        hint_path = find_file(cfg.layout_dir, image_name) or find_file(cfg.layout_dir, image_id)

        if img_path is None:
            missing_image += 1
            continue
        if hint_path is None:
            missing_hint += 1
            continue

        valid_rows.append(row)

    out = pd.DataFrame(valid_rows).reset_index(drop=True)
    print(f"[DATA] usable rows with image + hint: {len(out)}")
    print(f"[DATA] skipped rows: missing_image={missing_image}, missing_hint={missing_hint}")

    if len(out) == 0:
        raise RuntimeError("No usable samples found. Please check IMAGE_DIR, LAYOUT_DIR, and CSV columns.")

    return out, image_col


def select_reconstruction_rows(df: pd.DataFrame, cfg: InferenceConfig) -> pd.DataFrame:
    if cfg.max_recon is None or cfg.max_recon <= 0 or cfg.max_recon >= len(df):
        return df.copy().reset_index(drop=True)
    return df.sample(n=cfg.max_recon, random_state=cfg.seed).reset_index(drop=True)


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
    level_col = f"{dim}_level"
    if level_col in row.index and pd.notna(row[level_col]):
        return normalize_level(row[level_col])
    if dim in row.index and pd.notna(row[dim]):
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


def build_raw_score_prompt(row: pd.Series, cfg: InferenceConfig, override_scores: Optional[Dict[str, float]] = None) -> str:
    override_scores = override_scores or {}
    parts = []
    for dim in cfg.dimensions:
        sc = override_scores.get(dim, get_dim_score(row, dim))
        parts.append(f"{dim}: {sc:.2f}")

    core = "This is a street view image with human perception scores: " + ", ".join(parts) + "."
    core += " " + cfg.static_tail
    return core + quality_suffix(cfg)


def build_level_prompt(row: pd.Series, cfg: InferenceConfig, override_levels: Optional[Dict[str, str]] = None) -> str:
    override_levels = override_levels or {}
    parts = []
    for dim in cfg.dimensions:
        lv = override_levels.get(dim, get_dim_level(row, dim))
        lv = normalize_sweep_level(lv)
        parts.append(f"a {lv} level of {dim} perception")

    core = "This street view image targets " + ", ".join(parts) + "."
    core += " " + cfg.static_tail
    return core + quality_suffix(cfg)


def build_level_cue_prompt(row: pd.Series, cfg: InferenceConfig, override_levels: Optional[Dict[str, str]] = None) -> str:
    override_levels = override_levels or {}
    target_parts = []
    cue_sentences = []

    for dim in cfg.dimensions:
        lv = override_levels.get(dim, get_dim_level(row, dim))
        lv = normalize_sweep_level(lv)
        target_parts.append(f"a {lv} level of {dim} perception")

        attrs = DIM_ATTRS_EN.get(dim, [])
        if attrs:
            attr_parts = [f"a {lv} level of {a}" for a in attrs]
            cue_sentences.append(f"For {dim}, it has " + ", ".join(attr_parts) + ".")

    core = "This street view image targets " + ", ".join(target_parts) + "."
    if cue_sentences:
        core += " " + " ".join(cue_sentences)
    core += " " + cfg.static_tail
    return core + quality_suffix(cfg)


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

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0:
        print(f"[WARN] missing keys: {len(missing)}. First 10: {missing[:10]}")
    if len(unexpected) > 0:
        print(f"[WARN] unexpected keys: {len(unexpected)}. First 10: {unexpected[:10]}")

    exp_cfg = EXPERIMENTS[exp_name]
    model.use_view_transform_default = exp_cfg["use_view_transform"]

    model.to(device)
    model.eval()
    sampler = DDIMSampler(model)
    return model, sampler, ckpt_path


# ============================================================
# 6. Sampling
# ============================================================

def process_hint_image(hint_path: Union[str, Path], cfg: InferenceConfig, device: torch.device) -> torch.Tensor:
    m = cv2.imread(str(hint_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Cannot read hint image: {hint_path}")

    if m.ndim == 3 and m.shape[2] == 4:
        m = cv2.cvtColor(m, cv2.COLOR_BGRA2BGR)
    elif m.ndim == 2:
        m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

    control = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
    control = cv2.resize(control, (cfg.image_resolution, cfg.image_resolution), interpolation=cv2.INTER_AREA)
    control = torch.from_numpy(control).float().to(device) / 255.0
    control = torch.stack([control], dim=0)
    control = einops.rearrange(control, "b h w c -> b c h w")
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
        image_name = str(row["_image_name_raw"])
        image_id = str(row["_image_id"])

        img_path = find_file(cfg.image_dir, image_name) or find_file(cfg.image_dir, image_id)
        hint_path = find_file(cfg.layout_dir, image_name) or find_file(cfg.layout_dir, image_id)
        if img_path is None or hint_path is None:
            continue

        prompt = build_prompt_for_experiment(row, cfg, exp_name)
        out_img = gen_dir / f"{image_id}.png"

        if out_img.exists() and not cfg.overwrite:
            generated_paths = [out_img]
        else:
            try:
                control = process_hint_image(hint_path, cfg, device)
                sample_seed = cfg.seed + stable_int_hash(f"recon-{exp_name}-{image_id}")
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

    metadata_rows = []
    rows = select_sweep_rows(df, cfg)
    print(
        f"[SWEEP] {exp_name}: fixed {len(rows)} images × "
        f"{len(cfg.dimensions)} dims × {len(cfg.sweep_levels)} levels"
    )

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"sweep/{exp_name}"):
        image_name = str(row["_image_name_raw"])
        image_id = str(row["_image_id"])

        hint_path = find_file(cfg.layout_dir, image_name) or find_file(cfg.layout_dir, image_id)
        if hint_path is None:
            print(f"[SWEEP-SKIP] missing hint: {image_id}")
            continue

        try:
            control = process_hint_image(hint_path, cfg, device)
        except Exception as e:
            print(f"[SWEEP-SKIP] {exp_name}/{image_id}: {e}")
            continue

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

                if out_img.exists() and not cfg.overwrite:
                    generated_paths = [out_img]
                else:
                    try:
                        if cfg.same_latent_per_sweep:
                            sample_seed = cfg.seed + stable_int_hash(f"sweep-{exp_name}-{image_id}-{target_dim}")
                        else:
                            sample_seed = cfg.seed + stable_int_hash(
                                f"sweep-{exp_name}-{image_id}-{target_dim}-{level_word}"
                            )
                        set_global_seed(sample_seed)

                        imgs = sample_images(model, sampler, control, prompt, cfg)
                        generated_paths = save_generated_images(imgs, out_img, overwrite=cfg.overwrite)
                    except Exception as e:
                        print(f"[SWEEP-SKIP] {exp_name}/{image_id}/{target_dim}/{level_word}: {e}")
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
                        "source_hint_path": str(hint_path),
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
    main()

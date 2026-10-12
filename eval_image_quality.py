# -*- coding: utf-8 -*-
"""
Evaluate reconstruction image quality from folder structure, with incremental computation.

This script DOES NOT rely on metadata_all_both.csv or metadata_reconstruction.csv.
It builds pairs directly from:

    reconstruction/generated/<experiment>/<ckpt_tag>/<image_id>.png
    reconstruction/real/<image_id>.png

It computes:
  - LPIPS: paired perceptual distance, generated vs real, lower is better
  - SSIM: paired structural similarity, generated vs real, higher is better
  - FID: distribution distance, generated set vs real set, lower is better
  - KID: distribution distance, generated set vs real set, lower is better

Incremental behavior:
  - LPIPS/SSIM: already-computed generated-real pairs are skipped.
  - FID/KID: a group is skipped if its image set hash has not changed.
  - Set FORCE_RECOMPUTE=True to recompute everything.

Dependencies:
  pip install lpips scikit-image torchmetrics torch-fidelity pillow pandas tqdm
"""

from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


# ============================================================
# 0. USER SETTINGS: edit here only
# ============================================================

BASE_ROOT = Path("/home/yuebing/GenAI_density_urbanization/urban_data")


INFERENCE_ROOT = BASE_ROOT / "inference_outputs_untrained_baseline"
RECON_ROOT = INFERENCE_ROOT / "reconstruction"

REAL_DIR = RECON_ROOT / "real"
HINT_DIR = RECON_ROOT / "hint"
GENERATED_DIR = RECON_ROOT / "generated"

# 建议放在 inference_outputs_multidim_ablation 下面，便于和 reconstruction/sweep 并列管理
EVAL_ROOT= Path("/home/yuebing/GenAI_density_urbanization/eval")
OUTPUT_DIR = EVAL_ROOT / "evaluation_image_quality"/"untrained_baseline"

# 只评估这些 experiment；设为 None 表示扫描 generated 下所有 experiment
EXPERIMENTS: Optional[List[str]] = None
#["baseline", "view_only", "view_level_only", "full"]

# 只评估这些 ckpt_tag；设为 None 表示扫描所有 ckpt_tag
CKPT_TAGS: Optional[List[str]] = None
# CKPT_TAGS = ["epoch19-step135840", "epoch29-step203760"]

# 为了公平比较，是否只保留所有 experiment+ckpt 组共同拥有的 image_id
# 如果每个模型生成数量不完全一致，建议 True；如果只想把已有结果都算出来，可改 False
COMMON_IMAGE_IDS_ONLY= False

# 是否强制重算全部结果
FORCE_RECOMPUTE = False

# 如果同一路径图片被覆盖，是否把文件大小/修改时间纳入 FID/KID group_hash
# True 可以更好识别“同路径但图片被替换”的情况
HASH_INCLUDE_FILE_SIGNATURE = True

# 图像预处理
IMAGE_SIZE = 512
BATCH_SIZE = 24
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# LPIPS settings
LPIPS_NET = "alex"  # "alex" 常用且较快；可选 "vgg", "squeeze"

# FID/KID settings
FID_FEATURE = 2048
KID_SUBSET_SIZE = 100
KID_SUBSETS = 100

# 文件扩展名
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ============================================================
# 1. Paths and keys
# ============================================================

PAIR_KEY_COLS = ["experiment", "ckpt_tag", "image_id", "generated_path", "real_path"]

METADATA_USED_PATH = OUTPUT_DIR / "metadata_reconstruction_used.csv"
PAIRWISE_PATH = OUTPUT_DIR / "pairwise_lpips_ssim.csv"
PAIRWISE_SUMMARY_PATH = OUTPUT_DIR / "summary_pairwise_lpips_ssim.csv"
FID_KID_PATH = OUTPUT_DIR / "summary_fid_kid.csv"
SUMMARY_ALL_PATH = OUTPUT_DIR / "summary_image_quality_all_metrics.csv"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def strip_sample_suffix(stem: str) -> str:
    """
    If NUM_SAMPLES > 1, generated files may be like:
        imageid_s00.png, imageid_s01.png
    The corresponding real file is usually:
        imageid.png
    """
    return re.sub(r"_s\d+$", "", stem)


def find_same_stem_file(folder: Path, stem: str, preferred_ext: Optional[str] = None) -> Optional[Path]:
    """
    Find a file in folder with the given stem and common image extensions.
    Prefer preferred_ext if provided.
    """
    if preferred_ext:
        p = folder / f"{stem}{preferred_ext.lower()}"
        if p.exists() and p.is_file():
            return p

    for ext in sorted(IMAGE_EXTS):
        p = folder / f"{stem}{ext}"
        if p.exists() and p.is_file():
            return p
    return None


def make_key_series(df: pd.DataFrame, cols: Sequence[str] = PAIR_KEY_COLS) -> pd.Series:
    return df[list(cols)].astype(str).agg("||".join, axis=1)


def get_file_signature(path_str: str) -> str:
    """
    Small signature for detecting if a file at the same path has changed.
    It does not hash image content, only size + mtime, so it is fast.
    """
    try:
        p = Path(path_str)
        st = p.stat()
        return f"{p}|size={st.st_size}|mtime_ns={st.st_mtime_ns}"
    except Exception:
        return f"{path_str}|missing"


def compute_group_hash(group: pd.DataFrame) -> str:
    """
    Hash the exact image set used for one experiment+ckpt FID/KID group.
    If HASH_INCLUDE_FILE_SIGNATURE=True, file size and mtime are also included.
    """
    items = []
    g = group.sort_values(["image_id", "generated_path", "real_path"]).copy()
    for _, row in g.iterrows():
        if HASH_INCLUDE_FILE_SIGNATURE:
            gen_sig = get_file_signature(str(row["generated_path"]))
            real_sig = get_file_signature(str(row["real_path"]))
            items.append(f"{row['image_id']}||{gen_sig}||{real_sig}")
        else:
            items.append(f"{row['image_id']}||{row['generated_path']}||{row['real_path']}")
    text = "\n".join(items)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ============================================================
# 2. Build reconstruction pairs from folders
# ============================================================

def build_reconstruction_pairs_from_files() -> pd.DataFrame:
    """
    Build generated-real pairs from folder structure:

        GENERATED_DIR / <experiment> / <ckpt_tag> / <image_id>.png
        REAL_DIR / <image_id>.png

    Expected generated structure:
        reconstruction/generated/<experiment>/<ckpt_tag>/<image_id>.png

    This function does not use any CSV metadata.
    """
    if not GENERATED_DIR.exists():
        raise FileNotFoundError(f"Generated dir not found: {GENERATED_DIR}")
    if not REAL_DIR.exists():
        raise FileNotFoundError(f"Real dir not found: {REAL_DIR}")

    rows: List[Dict[str, str]] = []

    for gen_path in GENERATED_DIR.rglob("*"):
        if not gen_path.is_file():
            continue
        if gen_path.suffix.lower() not in IMAGE_EXTS:
            continue

        rel = gen_path.relative_to(GENERATED_DIR)
        parts = rel.parts

        # Expected: <experiment>/<ckpt_tag>/<image_file>
        if len(parts) < 3:
            print(f"[SKIP] unexpected generated path structure: {gen_path}")
            continue

        experiment = parts[0]
        ckpt_tag = parts[1]

        if EXPERIMENTS is not None and experiment not in EXPERIMENTS:
            continue
        if CKPT_TAGS is not None and ckpt_tag not in CKPT_TAGS:
            continue

        image_id = strip_sample_suffix(gen_path.stem)

        real_path = find_same_stem_file(REAL_DIR, image_id, preferred_ext=gen_path.suffix)
        if real_path is None:
            print(f"[MISSING REAL] image_id={image_id} for generated={gen_path}")
            continue

        hint_path = find_same_stem_file(HINT_DIR, image_id, preferred_ext=gen_path.suffix) if HINT_DIR.exists() else None

        rows.append(
            {
                "mode": "reconstruction",
                "experiment": experiment,
                "ckpt_tag": ckpt_tag,
                "image_id": image_id,
                "generated_filename": gen_path.name,
                "generated_path": str(gen_path),
                "real_path": str(real_path),
                "hint_path": str(hint_path) if hint_path is not None else "",
            }
        )

    meta = pd.DataFrame(rows)
    if meta.empty:
        raise RuntimeError(f"No valid generated-real pairs found under {GENERATED_DIR}")

    meta = meta.drop_duplicates(subset=PAIR_KEY_COLS).reset_index(drop=True)

    print("[PAIR BUILD] reconstruction pairs built from folders")
    print(f"  total pairs before common filtering: {len(meta)}")
    print("  by experiment / ckpt:")
    print(meta.groupby(["experiment", "ckpt_tag"]).size().to_string())

    return meta


def filter_common_image_ids(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only image_ids shared by every experiment+ckpt group.
    This makes comparison across models/checkpoints fair.
    """
    if not COMMON_IMAGE_IDS_ONLY:
        return meta.reset_index(drop=True)

    groups = list(meta.groupby(["experiment", "ckpt_tag"]))
    if len(groups) <= 1:
        return meta.reset_index(drop=True)

    common_ids: Optional[Set[str]] = None
    for _, g in groups:
        ids = set(g["image_id"].astype(str))
        common_ids = ids if common_ids is None else common_ids & ids

    common_ids = common_ids or set()
    before = len(meta)
    meta = meta[meta["image_id"].astype(str).isin(common_ids)].copy()

    print(
        f"[COMMON FILTER] common image_ids across {len(groups)} groups: {len(common_ids)}; "
        f"rows {before} -> {len(meta)}"
    )

    if meta.empty:
        raise RuntimeError(
            "COMMON_IMAGE_IDS_ONLY=True, but no common image_id exists across experiment+ckpt groups. "
            "Set COMMON_IMAGE_IDS_ONLY=False if you want to evaluate all available images."
        )

    print("  by experiment / ckpt after common filtering:")
    print(meta.groupby(["experiment", "ckpt_tag"]).size().to_string())

    return meta.reset_index(drop=True)


# ============================================================
# 3. Image loading
# ============================================================

def load_rgb_uint8(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def to_lpips_tensor(img: np.ndarray) -> torch.Tensor:
    # uint8 HWC [0,255] -> float CHW [-1,1]
    return torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0


def to_uint8_nchw(img: np.ndarray) -> torch.Tensor:
    # uint8 HWC [0,255] -> uint8 CHW [0,255]
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


# ============================================================
# 4. Incremental LPIPS / SSIM
# ============================================================

def split_pairwise_todo(meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      old_done: already computed rows that still belong to current meta
      todo: current meta rows not yet computed
    """
    if FORCE_RECOMPUTE or not PAIRWISE_PATH.exists():
        print("[INCREMENTAL] LPIPS/SSIM: recompute all or no previous pairwise file.")
        return pd.DataFrame(), meta.copy()

    old = pd.read_csv(PAIRWISE_PATH)

    required = PAIR_KEY_COLS + ["lpips", "ssim"]
    if not all(c in old.columns for c in required):
        print("[INCREMENTAL] old pairwise file columns incomplete; recompute all.")
        return pd.DataFrame(), meta.copy()

    current_keys = set(make_key_series(meta))
    old = old.copy()
    old["_key"] = make_key_series(old)

    old_done = old[
        old["_key"].isin(current_keys)
        & old["lpips"].notna()
        & old["ssim"].notna()
    ].copy()

    done_keys = set(old_done["_key"])

    meta2 = meta.copy()
    meta2["_key"] = make_key_series(meta2)
    todo = meta2[~meta2["_key"].isin(done_keys)].copy()

    old_done = old_done.drop(columns=["_key"]).copy()
    todo = todo.drop(columns=["_key"]).copy()

    print(
        f"[INCREMENTAL] LPIPS/SSIM: existing valid rows={len(old_done)}, "
        f"todo={len(todo)}, current total={len(meta)}"
    )

    return old_done, todo


def compute_pairwise_lpips_ssim(todo_meta: pd.DataFrame) -> pd.DataFrame:
    if todo_meta.empty:
        return pd.DataFrame()

    try:
        import lpips
    except Exception as e:
        raise ImportError("LPIPS package not found. Install with: pip install lpips") from e

    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception as e:
        raise ImportError("scikit-image not found. Install with: pip install scikit-image") from e

    device = torch.device(DEVICE)
    lpips_model = lpips.LPIPS(net=LPIPS_NET).to(device).eval()

    rows: List[Dict] = []
    batch_gen: List[torch.Tensor] = []
    batch_real: List[torch.Tensor] = []
    batch_indices: List[int] = []

    print("[PAIRWISE] Computing LPIPS and SSIM for todo rows...")
    for _, row in tqdm(todo_meta.iterrows(), total=len(todo_meta), desc="lpips_ssim"):
        out_row = row.to_dict()
        out_row["ssim"] = np.nan
        out_row["lpips"] = np.nan
        out_row["metric_error"] = ""

        try:
            gen = load_rgb_uint8(Path(row["generated_path"]))
            real = load_rgb_uint8(Path(row["real_path"]))

            # SSIM on RGB image.
            ssim_val = float(ssim(real, gen, channel_axis=2, data_range=255))
            out_row["ssim"] = ssim_val

            rows.append(out_row)
            rows_idx = len(rows) - 1

            batch_gen.append(to_lpips_tensor(gen))
            batch_real.append(to_lpips_tensor(real))
            batch_indices.append(rows_idx)

        except Exception as e:
            out_row["metric_error"] = repr(e)
            rows.append(out_row)
            continue

        if len(batch_gen) >= BATCH_SIZE:
            g = torch.stack(batch_gen).to(device)
            r = torch.stack(batch_real).to(device)
            with torch.no_grad():
                vals = lpips_model(g, r).view(-1).detach().cpu().numpy()
            for local_i, val in zip(batch_indices, vals):
                rows[local_i]["lpips"] = float(val)
            batch_gen, batch_real, batch_indices = [], [], []

    if batch_gen:
        g = torch.stack(batch_gen).to(device)
        r = torch.stack(batch_real).to(device)
        with torch.no_grad():
            vals = lpips_model(g, r).view(-1).detach().cpu().numpy()
        for local_i, val in zip(batch_indices, vals):
            rows[local_i]["lpips"] = float(val)

    return pd.DataFrame(rows)


def build_pairwise_results(meta: pd.DataFrame) -> pd.DataFrame:
    old_done, todo = split_pairwise_todo(meta)

    if len(todo) > 0:
        new_pairwise = compute_pairwise_lpips_ssim(todo)
        pairwise = pd.concat([old_done, new_pairwise], ignore_index=True)
    else:
        print("[INCREMENTAL] LPIPS/SSIM: no new rows to compute.")
        pairwise = old_done.copy()

    if pairwise.empty:
        raise RuntimeError("No pairwise LPIPS/SSIM results were produced.")

    # Keep only current meta keys and remove duplicates.
    current_keys = set(make_key_series(meta))
    pairwise["_key"] = make_key_series(pairwise)
    pairwise = pairwise[pairwise["_key"].isin(current_keys)].copy()
    pairwise = pairwise.drop_duplicates(subset=PAIR_KEY_COLS, keep="last")
    pairwise = pairwise.drop(columns=["_key"]).reset_index(drop=True)

    return pairwise


# ============================================================
# 5. Distribution metrics: FID / KID, with incremental group hash
# ============================================================

def batch_image_tensors(paths: Iterable[str], batch_size: int = BATCH_SIZE):
    batch: List[torch.Tensor] = []
    for p in paths:
        img = load_rgb_uint8(Path(p))
        batch.append(to_uint8_nchw(img))
        if len(batch) >= batch_size:
            yield torch.stack(batch)
            batch = []
    if batch:
        yield torch.stack(batch)


def compute_fid_kid_for_group(group: pd.DataFrame) -> Dict[str, float]:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception as e:
        raise ImportError(
            "torchmetrics image metrics not found. Install with: pip install torchmetrics torch-fidelity"
        ) from e

    device = torch.device(DEVICE)
    n = len(group)

    if n < 2:
        return {
            "fid": np.nan,
            "kid_mean": np.nan,
            "kid_std": np.nan,
            "kid_subset_size": np.nan,
            "kid_subsets": np.nan,
            "n_distribution": n,
        }

    fid_metric = FrechetInceptionDistance(feature=FID_FEATURE, normalize=False).to(device)

    # KID requires subset_size <= number of samples in both real and fake sets.
    subset_size = max(2, min(KID_SUBSET_SIZE, n))
    kid_metric = KernelInceptionDistance(
        subset_size=subset_size,
        subsets=KID_SUBSETS,
        normalize=False,
    ).to(device)

    for real_batch in batch_image_tensors(group["real_path"].tolist()):
        real_batch = real_batch.to(device)
        fid_metric.update(real_batch, real=True)
        kid_metric.update(real_batch, real=True)

    for fake_batch in batch_image_tensors(group["generated_path"].tolist()):
        fake_batch = fake_batch.to(device)
        fid_metric.update(fake_batch, real=False)
        kid_metric.update(fake_batch, real=False)

    with torch.no_grad():
        fid_val = float(fid_metric.compute().detach().cpu())
        kid_mean, kid_std = kid_metric.compute()
        kid_mean = float(kid_mean.detach().cpu())
        kid_std = float(kid_std.detach().cpu())

    return {
        "fid": fid_val,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "kid_subset_size": subset_size,
        "kid_subsets": KID_SUBSETS,
        "n_distribution": n,
    }


def load_existing_fid_kid() -> pd.DataFrame:
    if FORCE_RECOMPUTE or not FID_KID_PATH.exists():
        return pd.DataFrame()
    try:
        old = pd.read_csv(FID_KID_PATH)
    except Exception:
        return pd.DataFrame()

    required = ["experiment", "ckpt_tag", "group_hash", "fid", "kid_mean", "kid_std"]
    if not all(c in old.columns for c in required):
        return pd.DataFrame()
    return old


def compute_distribution_metrics_incremental(meta: pd.DataFrame) -> pd.DataFrame:
    old = load_existing_fid_kid()
    rows: List[Dict] = []

    print("[DISTRIBUTION] Computing or reusing FID/KID by experiment/checkpoint...")

    for (exp, ckpt), g in tqdm(meta.groupby(["experiment", "ckpt_tag"]), desc="fid_kid"):
        group_hash = compute_group_hash(g)
        reused = False

        if not old.empty:
            match = old[
                (old["experiment"].astype(str) == str(exp))
                & (old["ckpt_tag"].astype(str) == str(ckpt))
                & (old["group_hash"].astype(str) == str(group_hash))
                & old["fid"].notna()
                & old["kid_mean"].notna()
            ]
            if len(match) > 0:
                row = match.iloc[-1].to_dict()
                row["reused"] = True
                rows.append(row)
                reused = True

        if reused:
            continue

        vals = compute_fid_kid_for_group(g)
        vals.update(
            {
                "experiment": exp,
                "ckpt_tag": ckpt,
                "group_hash": group_hash,
                "reused": False,
            }
        )
        rows.append(vals)

    out = pd.DataFrame(rows)

    # Ensure a stable column order where possible.
    preferred = [
        "experiment", "ckpt_tag", "group_hash", "reused",
        "fid", "kid_mean", "kid_std", "kid_subset_size", "kid_subsets", "n_distribution",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols].reset_index(drop=True)


# ============================================================
# 6. Summaries
# ============================================================

def summarize_pairwise(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (exp, ckpt), g in pairwise.groupby(["experiment", "ckpt_tag"]):
        rows.append(
            {
                "experiment": exp,
                "ckpt_tag": ckpt,
                "n_pairs": len(g),
                "n_valid_lpips": int(g["lpips"].notna().sum()) if "lpips" in g.columns else 0,
                "n_valid_ssim": int(g["ssim"].notna().sum()) if "ssim" in g.columns else 0,
                "lpips_mean": g["lpips"].mean(),
                "lpips_std": g["lpips"].std(),
                "lpips_median": g["lpips"].median(),
                "ssim_mean": g["ssim"].mean(),
                "ssim_std": g["ssim"].std(),
                "ssim_median": g["ssim"].median(),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 7. Main
# ============================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    print(f"[INFO] DEVICE={DEVICE}")
    print(f"[INFO] INFERENCE_ROOT={INFERENCE_ROOT}")
    print(f"[INFO] GENERATED_DIR={GENERATED_DIR}")
    print(f"[INFO] REAL_DIR={REAL_DIR}")
    print(f"[INFO] OUTPUT_DIR={OUTPUT_DIR}")
    print(f"[INFO] COMMON_IMAGE_IDS_ONLY={COMMON_IMAGE_IDS_ONLY}")
    print(f"[INFO] FORCE_RECOMPUTE={FORCE_RECOMPUTE}")

    meta = build_reconstruction_pairs_from_files()
    meta = filter_common_image_ids(meta)

    meta.to_csv(METADATA_USED_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {METADATA_USED_PATH}")

    pairwise = build_pairwise_results(meta)
    pairwise.to_csv(PAIRWISE_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {PAIRWISE_PATH}")

    pairwise_summary = summarize_pairwise(pairwise)
    pairwise_summary.to_csv(PAIRWISE_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {PAIRWISE_SUMMARY_PATH}")

    dist = compute_distribution_metrics_incremental(meta)
    dist.to_csv(FID_KID_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {FID_KID_PATH}")

    summary = pairwise_summary.merge(
        dist.drop(columns=["reused"], errors="ignore"),
        on=["experiment", "ckpt_tag"],
        how="left",
    )
    summary.to_csv(SUMMARY_ALL_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {SUMMARY_ALL_PATH}")

    print("\n[DONE] Image quality evaluation completed.")
    print(summary.sort_values(["experiment", "ckpt_tag"]).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("default")
    main()
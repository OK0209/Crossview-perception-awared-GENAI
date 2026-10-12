
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


DEFAULT_DIMS = ["beautiful", "safety", "wealthy", "boring", "depressing", "lively"]
CLS_ORDER = ["low", "medium", "high"]


@dataclass
class BuildConfig:
    raw_csv: str
    output_csv: str
    output_threshold_json: str
    image_dir: str
    layout_dir: str
    image_col: str = "image_name"
    val_ratio: float = 0.1
    seed: int = 42
    std_k: float =1
    dimensions: Tuple[str, ...] = tuple(DEFAULT_DIMS)


def safe_exists_and_readable(path: str, flags: int) -> bool:
    if not os.path.exists(path):
        return False
    img = cv2.imread(path, flags)
    return img is not None


def resolve_paths(image_name: str, image_dir: str, layout_dir: str) -> Tuple[str, str, str, str]:
    image_name = str(image_name)
    image_file = image_name if image_name.endswith(".jpg") else f"{image_name}.jpg"
    hint_file = image_name if image_name.endswith(".png") else f"{image_name}.png"
    img_path = os.path.join(image_dir, image_file)
    hint_path = os.path.join(layout_dir, hint_file)
    return image_file, hint_file, img_path, hint_path


def classify_score(score: float, low_thr: float, high_thr: float) -> str:
    score = float(score)
    if score <= low_thr:
        return "low"
    if score >= high_thr:
        return "high"
    return "medium"


def compute_sample_weight(train_df: pd.DataFrame, dims: List[str]) -> pd.Series:
    """
    每个样本的权重 = 六个维度对应类别反频率权重的平均值
    """
    per_dim_weights: Dict[str, Dict[str, float]] = {}

    for dim in dims:
        counts = train_df[f"{dim}_cls"].value_counts().to_dict()
        total = sum(counts.get(k, 0) for k in CLS_ORDER)

        dim_weights = {}
        for cls_name in CLS_ORDER:
            cnt = counts.get(cls_name, 0)
            dim_weights[cls_name] = (total / cnt) if cnt > 0 else 0.0

        # 归一化到均值约为 1，避免量级过大
        vals = np.array([dim_weights[c] for c in CLS_ORDER], dtype=np.float64)
        vals = vals / max(vals.mean(), 1e-8)
        per_dim_weights[dim] = {c: float(v) for c, v in zip(CLS_ORDER, vals)}

    weights = []
    for _, row in train_df.iterrows():
        ws = [per_dim_weights[dim][row[f"{dim}_cls"]] for dim in dims]
        weights.append(float(np.mean(ws)))

    return pd.Series(weights, index=train_df.index, name="sample_weight")


def build_standardized_csv(cfg: BuildConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.raw_csv)
    dims = list(cfg.dimensions)

    required = [cfg.image_col] + dims
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in raw CSV: {missing}")

    valid_rows = []
    for _, row in df.iterrows():
        # 1) 六个维度都必须有合法分数
        if not all(pd.notna(row.get(dim)) and np.isfinite(float(row.get(dim))) for dim in dims):
            continue

        image_file, hint_file, img_path, hint_path = resolve_paths(
            row[cfg.image_col], cfg.image_dir, cfg.layout_dir
        )

        # 2) image / hint 必须存在且可读
        if not safe_exists_and_readable(img_path, cv2.IMREAD_COLOR):
            continue
        if not safe_exists_and_readable(hint_path, cv2.IMREAD_UNCHANGED):
            continue

        rec = row.to_dict()
        rec["image_file"] = image_file
        rec["hint_file"] = hint_file
        rec["img_path"] = img_path
        rec["hint_path"] = hint_path
        rec["is_valid"] = 1
        valid_rows.append(rec)

    clean_df = pd.DataFrame(valid_rows)
    if clean_df.empty:
        raise RuntimeError("No valid samples remain after filtering.")

    # 固定 split
    rng = np.random.default_rng(cfg.seed)
    indices = np.arange(len(clean_df))
    rng.shuffle(indices)
    val_size = int(round(len(clean_df) * cfg.val_ratio))
    val_idx = set(indices[:val_size].tolist())
    clean_df["split"] = ["val" if i in val_idx else "train" for i in range(len(clean_df))]

    train_df = clean_df[clean_df["split"] == "train"].copy()
    if train_df.empty:
        raise RuntimeError("Train split is empty. Check val_ratio.")

    thresholds = {}
    for dim in dims:
        mean_val = float(train_df[dim].mean())
        std_val = float(train_df[dim].std(ddof=0))

        low_thr = mean_val - cfg.std_k * std_val
        high_thr = mean_val + cfg.std_k * std_val

        thresholds[dim] = {
            "mean": mean_val,
            "std": std_val,
            "std_k": cfg.std_k,
            "low_thr": low_thr,
            "high_thr": high_thr,
        }

        clean_df[f"{dim}_cls"] = clean_df[dim].apply(
            lambda x: classify_score(x, low_thr, high_thr)
        )

    # 用同一份 train rows 生成同一套 sample weights，供全部实验共用
    train_weights = compute_sample_weight(clean_df[clean_df["split"] == "train"], dims)
    clean_df["sample_weight"] = 1.0
    clean_df.loc[train_weights.index, "sample_weight"] = train_weights.values

    # 记录固定顺序，便于 val prompt 完全可复现
    clean_df["fixed_dim_order"] = ",".join(dims)

    clean_df.to_csv(cfg.output_csv, index=False)
    with open(cfg.output_threshold_json, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)

    return clean_df


def get_default_config() -> BuildConfig:

    dir_my = "/home/yuebing/GenAI_density_urbanization/urban_data/"

    return BuildConfig(
        raw_csv=os.path.join(dir_my, "GSV_label_data.csv"),
        output_csv=os.path.join(dir_my, "GSV_label_data_standardized.csv"),
        output_threshold_json=os.path.join(dir_my, "standardization_thresholds.json"),
        image_dir=os.path.join(dir_my, "images_new"),
        layout_dir=os.path.join(dir_my, "layout_hint_colored_141_north"),
        image_col="image_name",
        val_ratio=0.1,
        seed=99,
        std_k=1,
        dimensions=tuple(DEFAULT_DIMS),
    )

def print_class_counts(df: pd.DataFrame, dims: List[str], split_name: str = "all"):
    print(f"\n===== Class counts ({split_name}) =====")
    for dim in dims:
        counts = df[f"{dim}_cls"].value_counts()
        ordered_counts = {cls_name: int(counts.get(cls_name, 0)) for cls_name in CLS_ORDER}
        print(f"{dim}: {ordered_counts}")

def main():
    cfg = get_default_config()
    out_df = build_standardized_csv(cfg)

    print(f"Saved standardized CSV to: {cfg.output_csv}")
    print(f"Saved thresholds JSON to: {cfg.output_threshold_json}")
    print(
        f"Valid samples: {len(out_df)} | "
        f"train={(out_df['split'] == 'train').sum()} | "
        f"val={(out_df['split'] == 'val').sum()}"
    )
    dims = list(cfg.dimensions)
    print_class_counts(out_df, dims, split_name="all")
    print_class_counts(out_df[out_df["split"] == "train"], dims, split_name="train")
    print_class_counts(out_df[out_df["split"] == "val"], dims, split_name="val")

if __name__ == "__main__":
    main()

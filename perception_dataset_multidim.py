import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_DIMS = ["beautiful", "safety", "wealthy", "boring", "depressing", "lively"]


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

CLS_WORD = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}


@dataclass
class PromptConfig:
    use_raw_score_prompt: bool = False
    use_level_prompt: bool = False
    use_level_cue_prompt: bool = False


class PerceptionMultiDimDataset(Dataset):
    def __init__(
        self,
        standardized_csv: str,
        image_dir: str,
        layout_dir: str,
        split: str,
        phase: str,
        dimensions: Optional[List[str]] = None,
        prompt_config: Optional[PromptConfig] = None,
        image_col: str = "image_file",
        hint_col: str = "hint_file",
        fixed_val_order: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.image_dir = image_dir
        self.layout_dir = layout_dir
        self.split = split
        self.phase = phase
        self.dimensions = dimensions or list(DEFAULT_DIMS)
        self.prompt_config = prompt_config or PromptConfig()
        self.image_col = image_col
        self.hint_col = hint_col
        self.fixed_val_order = fixed_val_order or list(self.dimensions)
        self.rng = random.Random(seed)

        df = pd.read_csv(standardized_csv)
        df = df[(df["is_valid"] == 1) & (df["split"] == split)].copy().reset_index(drop=True)
        if df.empty:
            raise RuntimeError(f"No samples found for split={split} in {standardized_csv}")
        self.df = df
        self.samples = df.to_dict(orient="records")

    def __len__(self):
        return len(self.samples)

    def _safe_imread(self, path: str, flags: int):
        img = cv2.imread(path, flags)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img

    def _load_target(self, img_path: str) -> np.ndarray:
        target = self._safe_imread(img_path, cv2.IMREAD_COLOR)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_LINEAR)
        target = (target.astype(np.float32) / 127.5) - 1.0
        return target

    def _load_hint(self, hint_path: str) -> np.ndarray:
        hint = self._safe_imread(hint_path, cv2.IMREAD_UNCHANGED)

        if hint.ndim == 2:
            hint = cv2.cvtColor(hint, cv2.COLOR_GRAY2BGR)
        elif hint.ndim == 3:
            if hint.shape[2] == 4:
                trans_mask = hint[:, :, 3] == 0
                if trans_mask.any():
                    hint[trans_mask] = [255, 255, 255, 255]
                hint = cv2.cvtColor(hint, cv2.COLOR_BGRA2BGR)
            elif hint.shape[2] > 4:
                hint = hint[:, :, :3]
        else:
            raise ValueError(f"Unexpected hint shape: {hint.shape}")

        hint = cv2.cvtColor(hint, cv2.COLOR_BGR2RGB)
        hint = cv2.resize(hint, (512, 512), interpolation=cv2.INTER_NEAREST)
        hint = hint.astype(np.float32) / 255.0
        return hint

    def _ordered_dims(self) -> List[str]:
        dims = list(self.dimensions)
        if self.phase == "train":
            self.rng.shuffle(dims)
            return dims
        return list(self.fixed_val_order)

    def _build_score_prompt(self, rec: Dict, dims: List[str]) -> str:
        parts = []
        tail = " Follow only static urban features (no people, no vehicles)."
        for dim in dims:
            score = float(rec[dim])
            parts.append(f"{dim} score of {score:.2f}")
        return "This is a street view with a human perception " + ", ".join(parts) + "."+ tail

    def _build_level_prompt(self, rec: Dict, dims: List[str]) -> str:
        parts = []
        tail = " Follow only static urban features (no people, no vehicles)."
        for dim in dims:
            cls_name = str(rec[f"{dim}_cls"])
            parts.append(f"a {CLS_WORD[cls_name]} level of {dim} perception")
        return "This street view image shows " + ", ".join(parts) + "." + tail


    def _build_level_cue_prompt(
        self,
        rec: Dict,
        dims: List[str],
    ) -> str:
        clauses = []

        for dim in dims:
            cls_name = str(rec[f"{dim}_cls"]).strip().lower()

            if cls_name not in CLS_WORD:
                raise ValueError(
                    f"Invalid class label for {dim}: {cls_name}"
                    )

            level = CLS_WORD[cls_name]
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

        return prompt
    


    def build_prompt(self, rec: Dict) -> str:
        dims = self._ordered_dims()
        cfg = self.prompt_config

        enabled = [
            cfg.use_raw_score_prompt,
            cfg.use_level_prompt,
            cfg.use_level_cue_prompt,
        ]
        if sum(bool(x) for x in enabled) != 1:
            raise ValueError("Exactly one prompt mode must be True.")

        if cfg.use_level_cue_prompt:
            return self._build_level_cue_prompt(rec, dims)
        if cfg.use_level_prompt:
            return self._build_level_prompt(rec, dims)
        return self._build_score_prompt(rec, dims)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        img_path = os.path.join(self.image_dir, rec[self.image_col])
        hint_path = os.path.join(self.layout_dir, rec[self.hint_col])

        target = self._load_target(img_path)
        hint = self._load_hint(hint_path)
        prompt = self.build_prompt(rec)

        sample = {
            "jpg": torch.from_numpy(target).contiguous(),
            "hint": torch.from_numpy(hint).contiguous(),
            "txt": prompt,
            "sample_weight": float(rec.get("sample_weight", 1.0)),
            "image_name": rec.get("image_name", os.path.splitext(rec[self.image_col])[0]),
        }
        for dim in self.dimensions:
            sample[f"{dim}_cls"] = str(rec[f"{dim}_cls"])
            sample[f"{dim}_score"] = float(rec[dim])
        return sample

import os
import random
from dataclasses import dataclass
from typing import Dict, List
from cldm.model import create_model, load_state_dict
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, WeightedRandomSampler

from perception_dataset_multidim import PerceptionMultiDimDataset, PromptConfig, DEFAULT_DIMS
from cldm.logger_view2_multi import ImageLogger


@dataclass
class TrainConfig:
    standardized_csv: str
    image_dir: str
    layout_dir: str
    experiment_name: str
    model_config_path: str
    resume_sd_path: str
    output_root: str
    batch_size: int = 4
    logger_freq: int = 500
    learning_rate: float = 1e-5
    sd_locked: bool = True
    only_mid_control: bool = False
    accumulate_grad_batches: int = 1
    precision: int = 32
    max_epochs: int = 30
    log_every_n_steps: int = 1000
    num_workers: int = 0
    seed: int = 42
    use_view_transform: bool = False
    use_level_prompt: bool = False
    use_level_cue_prompt: bool = False


EXPERIMENTS: Dict[str, Dict] = {
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


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def build_prompt_config(cfg: TrainConfig) -> PromptConfig:
    return PromptConfig(
        use_raw_score_prompt=not (cfg.use_level_prompt or cfg.use_level_cue_prompt),
        use_level_prompt=cfg.use_level_prompt,
        use_level_cue_prompt=cfg.use_level_cue_prompt,
    )


def build_datasets(cfg: TrainConfig):
    prompt_cfg = build_prompt_config(cfg)
    train_ds = PerceptionMultiDimDataset(
        standardized_csv=cfg.standardized_csv,
        image_dir=cfg.image_dir,
        layout_dir=cfg.layout_dir,
        split="train",
        phase="train",
        dimensions=list(DEFAULT_DIMS),
        prompt_config=prompt_cfg,
        seed=cfg.seed,
    )
    val_ds = PerceptionMultiDimDataset(
        standardized_csv=cfg.standardized_csv,
        image_dir=cfg.image_dir,
        layout_dir=cfg.layout_dir,
        split="val",
        phase="val",
        dimensions=list(DEFAULT_DIMS),
        prompt_config=prompt_cfg,
        seed=cfg.seed,
    )
    return train_ds, val_ds


def build_sampler(train_ds: PerceptionMultiDimDataset) -> WeightedRandomSampler:
    weights = torch.tensor([float(s["sample_weight"]) for s in train_ds.samples], dtype=torch.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class AblationImageLogger(ImageLogger):
    def __init__(self, use_view_transform: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_view_transform = use_view_transform

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if hasattr(pl_module, "use_view_transform_default"):
            pl_module.use_view_transform_default = self.use_view_transform
        return super().log_img(pl_module, batch, batch_idx, split)

    
def get_default_config() -> TrainConfig:
    dir_my = "/home/yuebing/GenAI_density_urbanization/urban_data/"
    output_root = "./ckpts_s/checkpoints_perception"

    return TrainConfig(
        standardized_csv=os.path.join(dir_my, "GSV_label_data_standardized.csv"),
        image_dir=os.path.join(dir_my, "images_new"),
        layout_dir=os.path.join(dir_my, "layout_hint_colored_141_north"),
        experiment_name="full",   # 这里改实验组
        model_config_path="./models/cldm_v15_unified.yaml",
        resume_sd_path="./models/control_sd15_ini.ckpt",
        output_root=output_root,
        batch_size=4,
        logger_freq=500,
        learning_rate=1e-5,
        accumulate_grad_batches=1,
        precision=32,
        max_epochs=30,
        log_every_n_steps=1000,
        num_workers=0,
        seed=42,
    )

def run_one_experiment(cfg: TrainConfig):
    set_global_seed(cfg.seed)

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"train_{cfg.experiment_name}.log")
    log_file = open(log_path, "a", encoding="utf-8")

    import sys
    old_stdout, old_stderr = sys.stdout, sys.stderr


    try:
        print(f"[EXP] {cfg.experiment_name}")
        print(
            f"[CFG] use_view_transform={cfg.use_view_transform}, "
            f"use_level_prompt={cfg.use_level_prompt}, "
            f"use_level_cue_prompt={cfg.use_level_cue_prompt}"
        )

        train_ds, val_ds = build_datasets(cfg)
        sampler = build_sampler(train_ds)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        model = create_model(cfg.model_config_path).cpu()

        exp_dir = os.path.join(cfg.output_root, cfg.experiment_name, "multi")
        last_ckpt = os.path.join(exp_dir, "last.ckpt")

        if not os.path.exists(last_ckpt):
            model.load_state_dict(load_state_dict(cfg.resume_sd_path, location="cpu"), strict=False)
        model.learning_rate = cfg.learning_rate
        model.sd_locked = cfg.sd_locked
        model.only_mid_control = cfg.only_mid_control
        model.use_view_transform_default = cfg.use_view_transform

        exp_dir = os.path.join(cfg.output_root, cfg.experiment_name, "multi")
        os.makedirs(exp_dir, exist_ok=True)

        checkpoint_callbacks = [
            ModelCheckpoint(
                dirpath=exp_dir,
                filename="epoch{epoch}-step{step}",
                every_n_epochs=5,
                save_top_k=-1,
                auto_insert_metric_name=False,
            ),
            ModelCheckpoint(
                dirpath=exp_dir,
                filename="last",
                save_last=True,
                every_n_train_steps=1000,
                save_top_k=1,
                save_on_train_epoch_end=False,
            ),
            ModelCheckpoint(
                dirpath=exp_dir,
                filename="best-{epoch}-{step}",
                monitor="loss_simple_ema",
                mode="min",
                save_top_k=1,
                auto_insert_metric_name=False,
            ),
        ]

        img_logger = AblationImageLogger(
            use_view_transform=cfg.use_view_transform,
            batch_frequency=cfg.logger_freq,
            run_name=cfg.experiment_name,
            save_keys=["proj_vis", "bev_input", "reconstruction", "control"],
        )

        tb_logger = TensorBoardLogger(
            save_dir="/home/yuebing/GenAI_density_urbanization/tb_logs",
            name="multidim_ablation",
            version=cfg.experiment_name,
        )

        accelerator = "gpu" if torch.cuda.is_available() else "auto"
        devices = [0] if torch.cuda.is_available() else 1

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=cfg.precision,
            accumulate_grad_batches=cfg.accumulate_grad_batches,
            callbacks=[img_logger] + checkpoint_callbacks,
            logger=tb_logger,
            check_val_every_n_epoch=1,
            max_epochs=cfg.max_epochs,
            log_every_n_steps=cfg.log_every_n_steps,
            enable_progress_bar=True,
        )

        trainer.fit(
            model,
            train_loader,
            val_loader,
            ckpt_path=last_ckpt if os.path.exists(last_ckpt) else None,
        )

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()
def main():
    base_cfg = get_default_config()

    experiment_order = [

        "baseline"
    ]#        "full", "view_level_only", "view_only",

    for exp_name in experiment_order:
        exp_cfg = EXPERIMENTS[exp_name]

        cfg = TrainConfig(**base_cfg.__dict__)
        cfg.experiment_name = exp_name
        cfg.use_view_transform = exp_cfg["use_view_transform"]
        cfg.use_level_prompt = exp_cfg["use_level_prompt"]
        cfg.use_level_cue_prompt = exp_cfg["use_level_cue_prompt"]


        print(f"\n========== START {exp_name} ==========\n")
        run_one_experiment(cfg)
        print(f"\n========== END {exp_name} ==========\n")
if __name__ == "__main__":
    main()


#logger.py
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, run_name="default"):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.run_name = run_name   # [MOD] 不同训练类型对应不同子文件夹

    @staticmethod
    def _sanitize_name(x: str) -> str:
        x = str(x)
        for ch in ['/', '\\', ' ', ':', ';', ',', '[', ']', '(', ')', '{', '}', "'"]:
            x = x.replace(ch, '-')
        return x

    @staticmethod
    def _build_batch_name(batch, max_images: int):
        """
        从 batch 中取图片名，拼成：
        name1__name2__name3__name4
        """
        names = None

        if isinstance(batch, dict):
            if "image_name" in batch:
                names = batch["image_name"]
            elif "image_file" in batch:
                names = batch["image_file"]

        if names is None:
            return "unknown"

        # DataLoader collate 后通常是 list[str]
        if isinstance(names, (list, tuple)):
            names = list(names)[:max_images]
            names = [os.path.splitext(str(x))[0] for x in names]
            names = [ImageLogger._sanitize_name(x) for x in names]
            return "__".join(names)

        # 兜底
        return ImageLogger._sanitize_name(str(names))

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx, batch_name):
        # [MOD] 增加 run_name 子文件夹
        root = os.path.join(save_dir, "image_log", split, self.run_name)

        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1

            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)

            # [MOD] 文件名改成带输入图片名
            filename = "{}_{}_e-{:02}_b-{:05}.png".format(
                k, batch_name, current_epoch, batch_idx
            )

            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx
        if (self.check_frequency(check_idx) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            # [MOD] 只保留 control 的话可打开这段
            if "control" in images:
                images = {"control": images["control"]}
            else:
                if is_train:
                    pl_module.train()
                return

            batch_name = self._build_batch_name(batch, self.max_images)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                batch_name
            )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
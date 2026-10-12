import os
import re
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency=2000,
        max_images=4,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
        run_name="default",
        save_keys=None,
    ):
        """
        Args:
            batch_frequency: 每隔多少个 batch_idx 记录一次
            max_images: 每次最多保存多少张
            clamp: 是否先 clamp 到 [-1,1]
            rescale: 是否把 [-1,1] -> [0,1]
            run_name: image_log/train/<run_name>/...
            save_keys:
                None  -> 自动按 workflow2 推荐顺序挑选存在的 key
                list  -> 只保存这些 key，例如 ["proj_vis", "control_feat_vis"]
        """
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.run_name = run_name
        self.save_keys = save_keys

        # 兼容旧参数
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------
    @staticmethod
    def _sanitize_name(x: str) -> str:
        x = str(x)
        x = os.path.splitext(x)[0]
        x = re.sub(r"[\/\\\s:;,\[\]\(\)\{\}'\"|<>?*]+", "-", x)
        x = re.sub(r"-+", "-", x).strip("-")
        return x if x else "unknown"

    @staticmethod
    def _build_batch_name(batch, max_images: int):
        """
        从 batch 中取图片名，拼成:
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
        all_dims = ["beautiful", "safety", "lively", "wealthy", "depressing", "boring"]
        if isinstance(names, (list, tuple)):
            names = list(names)[:max_images]
            batch_parts = []
            
            for i, name in enumerate(names):
                base_id = ImageLogger._sanitize_name(name)[:8] # [技巧] 只取 ID 前 8 位缩短长度
                
                # 2. 提取 6 位数字码
                level_code = ""
                cls_map = {"low": "0", "medium": "1", "high": "2"}

                for dim in all_dims:
                    key = f"{dim}_cls"
                    if key in batch:
                        val = batch[key][i]
                        if torch.is_tensor(val):
                            val = val.item()
                        level_code += cls_map.get(str(val), "x")
                
                # 拼接：ID前缀_6位码
                batch_parts.append(f"{base_id}_{level_code}")
            
            return "__".join(batch_parts)

        return ImageLogger._sanitize_name(names)

    @staticmethod
    def _select_image_keys(images: dict, save_keys=None):
        """
        workflow2 推荐优先级：
        proj_vis           真正几何投影视觉图
        control_feat_vis   feature 可视化
        bev_input          原始 BEV
        reconstruction     VAE reconstruction
        conditioning       文本条件图
        samples            采样图
        control            兼容旧版本
        """
        if save_keys is not None:
            return [k for k in save_keys if k in images]

        preferred = [
            "proj_vis",
            "control_feat_vis",
            "bev_input",
            "reconstruction",
            "conditioning",
            "samples",
            "control",
        ]
        selected = [k for k in preferred if k in images]

        # 如果一个都没选到，就退化成全部 tensor 图
        if len(selected) == 0:
            for k, v in images.items():
                if isinstance(v, torch.Tensor):
                    selected.append(k)

        return selected

    @staticmethod
    def _prepare_single_tensor(x: torch.Tensor, clamp=True, rescale=True):
        """
        输入:
            x: [N,C,H,W]
        输出:
            适合 torchvision.make_grid 的 CPU tensor
        """
        x = x.detach().cpu()

        if clamp:
            x = torch.clamp(x, -1.0, 1.0)

        # 如果开启 rescale，则按 [-1,1] -> [0,1]
        if rescale:
            x = (x + 1.0) / 2.0

        # 最后兜底 clamp 到 [0,1]
        x = torch.clamp(x, 0.0, 1.0)
        return x

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx, batch_name):
        root = os.path.join(
            save_dir,
            "image_log",
            split,
            self.run_name,
            f"epoch_{current_epoch:03d}"
        )
        os.makedirs(root, exist_ok=True)

        for k, v in images.items():
            if not isinstance(v, torch.Tensor):
                continue

            grid = torchvision.utils.make_grid(v, nrow=min(self.max_images, v.shape[0]))
            grid = grid.permute(1, 2, 0).numpy()
            grid = np.clip(grid * 255.0, 0, 255).astype(np.uint8)

            filename = "{}_{}_e-{:06d}_b-{:06d}.png".format(
                k, batch_name, current_epoch, batch_idx
            )
            path = os.path.join(root, filename)
            Image.fromarray(grid).save(path)

    # --------------------------------------------------
    # main logging
    # --------------------------------------------------
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step

        if not (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            return

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        try:
            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
        except Exception as e:
            print(f"[ImageLogger] log_images failed: {e}")
            if is_train:
                pl_module.train()
            return

        if not isinstance(images, dict) or len(images) == 0:
            if is_train:
                pl_module.train()
            return

        selected_keys = self._select_image_keys(images, self.save_keys)
        if len(selected_keys) == 0:
            if is_train:
                pl_module.train()
            return

        batch_name = self._build_batch_name(batch, self.max_images)

        processed = {}
        for k in selected_keys:
            v = images[k]
            if not isinstance(v, torch.Tensor):
                continue
            # 如果通道数不是 3 (比如是 16)，强制只取前 3 个通道，防止 stack 报错
            if v.shape[1] != 3:
                print(f"⚠️ [ImageLogger] 警告: {k} 通道数为 {v.shape[1]}，已自动截取前 3 通道进行可视化")
                v = v[:, :3, :, :]
                
            # 取前 max_images 张
            N = min(v.shape[0], self.max_images)
            v = v[:N]

            # 统一处理到 [0,1]
            v = self._prepare_single_tensor(v, clamp=self.clamp, rescale=self.rescale)
            processed[k] = v

        if len(processed) > 0:
            self.log_local(
                pl_module.logger.save_dir,
                split,
                processed,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                batch_name
            )

        if is_train:
            pl_module.train()

    def check_frequency(self, check_idx):
        if self.log_first_step and check_idx == 0:
            return True
        return (check_idx % self.batch_freq) == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
import einops
import torch
import torch as th
import torch.nn as nn
from cldm.geometry_projector import IdentityProjector

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    AttentionBlock,
)
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for _, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert use_spatial_transformer, "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)

        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks)))
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels

                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []
        h = x.type(self.dtype)

        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
        return outs


class ControlLDM(LatentDiffusion):
    def __init__(self, control_stage_config, control_key, only_mid_control,
                 geometry_projector_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        if geometry_projector_config is not None:
            self.geometry_projector = instantiate_from_config(geometry_projector_config)
        else:
            self.geometry_projector = IdentityProjector(out_size=512)

    def project_control_hint(self, control):
        """
        control: [B, C, H, W], raw BEV hint
        return:  [B, C', H', W'], projected view hint / feature
        """
        return self.geometry_projector(control)
    
    def _prepare_middle_inputs(self, x, t, c):
            """
            核心优化点：在中间层提取时也支持 control_is_projected 标志。
            """
            cond_txt = torch.cat(c["c_crossattn"], 1)
            raw_or_proj_control = torch.cat(c["c_concat"], 1)

            # 🌟 修改：检查是否需要重复投影
            if c.get("control_is_projected", False):
                projected_control = raw_or_proj_control
            else:
                projected_control = self.project_control_hint(raw_or_proj_control)

            control = self.control_model(x=x, hint=projected_control, timesteps=t, context=cond_txt)
            control = [c_i * scale for c_i, scale in zip(control, self.control_scales)]

            diffusion_model = self.model.diffusion_model
            t_emb = timestep_embedding(t, diffusion_model.model_channels, repeat_only=False)
            emb = diffusion_model.time_embed(t_emb)
            return cond_txt, projected_control, control, emb
    
    @torch.no_grad()
    def inspect_unet_feature_shapes(self, x, t, c):
        cond_txt = torch.cat(c["c_crossattn"], 1)
        diffusion_model = self.model.diffusion_model

        t_emb = timestep_embedding(t, diffusion_model.model_channels, repeat_only=False)
        emb = diffusion_model.time_embed(t_emb)

        h = x.type(diffusion_model.dtype)
        shapes = []

        for i, module in enumerate(diffusion_model.input_blocks):
            h = module(h, emb, cond_txt)
            shapes.append(("input_blocks", i, tuple(h.shape)))

        h = diffusion_model.middle_block(h, emb, cond_txt)
        shapes.append(("middle_block", 0, tuple(h.shape)))
        return shapes
    
    @torch.no_grad()
    def extract_base_middle_embedding(self, x, t, c):
        """
        Return the base diffusion middle embedding:
        depends on x + timestep + prompt
        but WITHOUT injected control residual.
        Output shape: [B, C, H, W]
        """
        cond_txt = torch.cat(c["c_crossattn"], 1)
        diffusion_model = self.model.diffusion_model

        t_emb = timestep_embedding(t, diffusion_model.model_channels, repeat_only=False)
        emb = diffusion_model.time_embed(t_emb)

        h = x.type(diffusion_model.dtype)
        for module in diffusion_model.input_blocks:
            h = module(h, emb, cond_txt)
        h = diffusion_model.middle_block(h, emb, cond_txt)
        return h
    
    @torch.no_grad()
    def extract_layout_only_middle_embedding(self, x, t, c):
        """
        Return layout-only controlled middle embedding:
        depends on x + timestep + layout control
        but WITHOUT text prompt.
        Output shape: [B, C, H, W]
        """
        batch_size = x.shape[0]
        uncond_c_txt = self.get_unconditional_conditioning(batch_size)

        c_layout_only = {
            "c_crossattn": [uncond_c_txt],
            "c_concat": c["c_concat"],
        }

        if c.get("control_is_projected", False):
            c_layout_only["control_is_projected"] = True

        return self.extract_controlled_middle_embedding(x, t, c_layout_only)
    
    @torch.no_grad()
    def extract_controlled_middle_embedding(self, x, t, c):
        # 🌟 这里的逻辑已经通过修改 _prepare_middle_inputs 得到了加速
        cond_txt, _, control, emb = self._prepare_middle_inputs(x, t, c)
        diffusion_model = self.model.diffusion_model
        h = x.type(diffusion_model.dtype)
        for module in diffusion_model.input_blocks:
            h = module(h, emb, cond_txt)
        h = diffusion_model.middle_block(h, emb, cond_txt)
        if control is not None:
            h = h + control.pop()
        return h

    @torch.no_grad()
    def extract_control_delta(self, x, t, c):
        """
        Return delta = controlled - base
        Output shape: [B, C, H, W]
        """
        base = self.extract_base_middle_embedding(x, t, c)
        controlled = self.extract_controlled_middle_embedding(x, t, c)
        return controlled - base


    @torch.no_grad()
    def extract_middle_embedding_bundle(self, x, t, c, do_extract=True):
        if not do_extract:
            return {}

        # 如果传入的是原始 BEV，在这里只投影一次供后续复用
        if not c.get("control_is_projected", False):
            raw_control = torch.cat(c["c_concat"], 1)
            projected_control = self.project_control_hint(raw_control)
            c_for_extract = c.copy()
            c_for_extract["c_concat"] = [projected_control]
            c_for_extract["control_is_projected"] = True
        else:
            c_for_extract = c

        batch_size = x.shape[0]
        uncond_c_txt = self.get_unconditional_conditioning(batch_size)
        c_null = {
            "c_crossattn": [uncond_c_txt],
            "c_concat": None
        }

        # 1) base: 无 text，无 layout
        base_map = self.extract_base_middle_embedding(x, t, c_null)

        # 2) layout-only: 无 text，有 layout
        layout_controlled_map = self.extract_layout_only_middle_embedding(x, t, c_for_extract)

        # 3) full(text on top of layout): 有 text，有 layout
        text_controlled_map = self.extract_controlled_middle_embedding(x, t, c_for_extract)

        results = {
            "base_map": base_map,
            "layout_controlled_map": layout_controlled_map,
            "text_controlled_map": text_controlled_map,
            "layout_delta_map": layout_controlled_map - base_map,
            "text_delta_map": text_controlled_map - layout_controlled_map,
        }

        for name in ["base", "layout_controlled", "text_controlled", "layout_delta", "text_delta"]:
            m = results[f"{name}_map"]
            results[f"{name}_vec_avg"] = m.mean(dim=(2, 3))
            results[f"{name}_vec_max"] = m.amax(dim=(2, 3))

        return results

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, "b h w c -> b c h w")
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond["c_crossattn"], 1)
        if cond["c_concat"] is None:
            return diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        
        raw_or_proj_control = torch.cat(cond["c_concat"], 1)
        # 🌟 已有优化：推理循环中跳过重复投影
        if cond.get("control_is_projected", False):
            projected_control = raw_or_proj_control
        else:
            projected_control = self.project_control_hint(raw_or_proj_control)

        control = self.control_model(x=x_noisy, hint=projected_control, timesteps=t, context=cond_txt)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        return diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

    @torch.no_grad()
    def visualize_control_projection(self, control):
        """
        control: raw BEV [B,C,H,W]
        return: RGB visualization [B,3,H,W] in [0,1]
        """
        if hasattr(self.geometry_projector, "visualize_projection"):
            return self.geometry_projector.visualize_projection(control)
        else:
            feat = self.project_control_hint(control)
            vis = feat[:, :3]
            if vis.shape[1] < 3:
                rep = 3 // vis.shape[1] + 1
                vis = vis.repeat(1, rep, 1, 1)[:, :3]
            vmin = vis.amin(dim=(1, 2, 3), keepdim=True)
            vmax = vis.amax(dim=(1, 2, 3), keepdim=True)
            vis = (vis - vmin) / (vmax - vmin + 1e-6)
            return vis

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=4,
        n_row=2,
        sample=False,
        ddim_steps=50,
        ddim_eta=0.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=True,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=False,
        unconditional_guidance_scale=9.0,
        unconditional_guidance_label=None,
        use_ema_scope=True,
        **kwargs
    ):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)

        log["reconstruction"] = self.decode_first_stage(z)

        bev_vis = c_cat[:, :3]
        if bev_vis.shape[1] < 3:
            rep = 3 // bev_vis.shape[1] + 1
            bev_vis = bev_vis.repeat(1, rep, 1, 1)[:, :3]
        log["bev_input"] = bev_vis * 2.0 - 1.0

        proj_vis = self.visualize_control_projection(c_cat)
        log["proj_vis"] = proj_vis * 2.0 - 1.0

        projected_cat = self.project_control_hint(c_cat)
        feat_vis = projected_cat[:, :3]
        if feat_vis.shape[1] < 3:
            rep = 3 // feat_vis.shape[1] + 1
            feat_vis = feat_vis.repeat(1, rep, 1, 1)[:, :3]
        vmin = feat_vis.amin(dim=(1, 2, 3), keepdim=True)
        vmax = feat_vis.amax(dim=(1, 2, 3), keepdim=True)
        feat_vis = (feat_vis - vmin) / (vmax - vmin + 1e-6)
        log["control_feat_vis"] = feat_vis * 2.0 - 1.0

        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t_cur = repeat(torch.tensor([t]), "1 -> b", b=n_row).to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t_cur, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)
            diffusion_grid = rearrange(diffusion_row, "n b c h w -> b n c h w")
            diffusion_grid = rearrange(diffusion_grid, "b n c h w -> (b n) c h w")
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            samples, z_denoise_row = self.sample_log(
                cond={"c_concat": [c_cat], "c_crossattn": [c]},
                batch_size=N,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta
            )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(
                cond={"c_concat": [c_cat], "c_crossattn": [c]},
                batch_size=N,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
            )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)

        control_tensor = cond["c_concat"][0]
        if cond.get("control_is_projected", False):
            projected_control = control_tensor
        else:
            projected_control = self.project_control_hint(control_tensor)

        cond_proj = {
            "c_concat": [projected_control],
            "c_crossattn": cond["c_crossattn"],
            "control_is_projected": True
        }

        _, _, h, w = projected_control.shape
        shape = (self.channels, h // 8, w // 8)

        samples, intermediates = ddim_sampler.sample(
            ddim_steps, batch_size, shape, cond_proj, verbose=False, **kwargs
        )
        return samples, intermediates

    def configure_optimizers(self):
        params = list(self.control_model.parameters())
        if hasattr(self, "geometry_projector"):
            params += list(self.geometry_projector.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        return torch.optim.AdamW(params, lr=self.learning_rate)

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.geometry_projector = self.geometry_projector.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.geometry_projector = self.geometry_projector.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
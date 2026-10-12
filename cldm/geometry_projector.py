# cldm/geometry_projector.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseGeometryProjector(nn.Module):
    def forward(self, bev_hint: torch.Tensor) -> torch.Tensor:
        """
        bev_hint: [B, C, H, W], raw BEV hint in [0,1], RGB
        return:   [B, F, H_out, W_out], projected feature map
        """
        raise NotImplementedError


class IdentityProjector(BaseGeometryProjector):
    def __init__(self, out_size=512, out_channels=16):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.Conv2d(3, out_channels, kernel_size=1)

    def forward(self, bev_hint: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(bev_hint, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return self.proj(x)


class FixedNorthBEVProjector(BaseGeometryProjector):
    ROAD_HEX = {
        0: "#fef0d9",
        1: "#ffd8a1",
        2: "#ffb74d",
        3: "#ff8f00",
        4: "#ff6d00",
        5: "#e65100",
        6: "#bf360c",
    }

    BLD_BINS = {
        "lt10":   {"hex": ["#e0f2ff", "#d0e8ff"], "h_center": 10.0},
        "10_20":  {"hex": ["#bbdefb"],            "h_center": 20.0},
        "20_40":  {"hex": ["#90caf9"],            "h_center": 50.0},
        "40_60":  {"hex": ["#64b5f6"],            "h_center": 80.0},
        "60_100": {"hex": ["#42a5f5"],            "h_center": 120.0},
        "ge100":  {"hex": ["#1e88e5"],            "h_center": 150.0},
    }

    def __init__(
        self,
        out_size=512,
        bev_meters=282.0,
        hfov=70.0,
        cam_h=1.7,
        pitch=0.0,
        t_far=285.0,
        step=0.2,
        tol_road=22,
        tol_bld=18,
        cam_clear_r=6,
        white_thr=25,
        feature_channels=16,
        hidden_channels=32,
    ):
        super().__init__()
        self.out_size = int(out_size)
        self.bev_meters = float(bev_meters)
        self.hfov = float(hfov)
        self.cam_h = float(cam_h)
        self.pitch = float(pitch)
        self.t_far = float(t_far)
        self.step = float(step)
        self.tol_road = float(tol_road)
        self.tol_bld = float(tol_bld)
        self.cam_clear_r = int(cam_clear_r)
        self.white_thr = float(white_thr)
        self.register_buffer(
            "road_rgb_color",
            torch.tensor([255.0, 140.0, 0.0], dtype=torch.float32) / 255.0,
            persistent=False
        )
        self.register_buffer(
            "bld_rgb_color",
            torch.tensor([0.0, 0.0, 255.0], dtype=torch.float32) / 255.0,
            persistent=False
        )

        road_rgb = []
        for lv in sorted(self.ROAD_HEX.keys()):
            road_rgb.append(self.hex_to_rgb(self.ROAD_HEX[lv]))
        road_rgb = torch.tensor(road_rgb, dtype=torch.float32)

        bld_rgb = []
        bld_h = []
        for _, info in self.BLD_BINS.items():
            for hx in info["hex"]:
                bld_rgb.append(self.hex_to_rgb(hx))
                bld_h.append(float(info["h_center"]))
        bld_rgb = torch.tensor(bld_rgb, dtype=torch.float32)
        bld_h = torch.tensor(bld_h, dtype=torch.float32)

        self.register_buffer("road_rgb_bank", road_rgb, persistent=False)
        self.register_buffer("bld_rgb_bank", bld_rgb, persistent=False)
        self.register_buffer("bld_height_bank", bld_h, persistent=False)

        # [MOD] Workflow2: geometry features -> learned feature embedding
        # input channels:
        #   0 depth_norm
        #   1 hit_bld
        #   2 hit_road
        #   3 hit_ground
        self.feature_head = nn.Sequential(
            nn.Conv2d(4, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, feature_channels, kernel_size=3, padding=1),
        )

    @staticmethod
    def hex_to_rgb(hexstr: str):
        s = hexstr.strip().lstrip("#")
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return [r, g, b]

    def mask_to_morph_edges_torch(self, mask: torch.Tensor, k=3):
        x = mask.unsqueeze(1).float()
        dil = self._dilate(x, k)
        ero = self._erode(x, k)
        edge = (dil - ero) > 0
        return edge[:, 0]

    def compose_vis_torch(self, depth_m, hit_bld, hit_road):
        device = depth_m.device
        dtype = depth_m.dtype
        B, H, W = depth_m.shape

        hit_road2 = self._dilate(hit_road.unsqueeze(1).float(), k=5)[:, 0] > 0.5
        edges = self.mask_to_morph_edges_torch(hit_bld, k=3)

        m_bld = hit_bld
        m_road = hit_road2
        mask = m_bld | m_road

        depth01 = 1.0 - torch.clamp(depth_m / float(self.t_far), 0.0, 1.0)

        Dn = torch.zeros_like(depth01)
        for b in range(B):
            v = depth01[b][mask[b]]
            if v.numel() > 0:
                vmin = torch.quantile(v, 0.05)
                vmax = torch.quantile(v, 0.95)
                Dn[b] = torch.clamp((depth01[b] - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
            else:
                Dn[b] = depth01[b]

        min_bright = 0.25
        bright = min_bright + (1.0 - min_bright) * Dn

        vis = torch.zeros((B, 3, H, W), device=device, dtype=dtype)

        road_color = self.road_rgb_color.to(device=device, dtype=dtype).view(1, 3, 1, 1)
        bld_color = self.bld_rgb_color.to(device=device, dtype=dtype).view(1, 3, 1, 1)

        vis = torch.where(m_road.unsqueeze(1), road_color * bright.unsqueeze(1), vis)
        vis = torch.where(m_bld.unsqueeze(1), bld_color * bright.unsqueeze(1), vis)

        vis = torch.where(edges.unsqueeze(1), torch.ones_like(vis), vis)
        return vis

    @torch.no_grad()
    def visualize_projection(self, bev_hint: torch.Tensor) -> torch.Tensor:
        bev_hint = bev_hint.float()

        height_map_m, bld_any, road_any = self.parse_bev_torch(bev_hint)
        road_mask_bev = road_any & (~bld_any)

        depth_m, hit_bld, hit_road, hit_ground = self.render_view_from_bev_torch(
            height_map_m=height_map_m,
            road_mask_bev=road_mask_bev,
        )

        vis = self.compose_vis_torch(depth_m, hit_bld, hit_road)
        return vis
    @staticmethod
    def build_camera_intrinsics(img_w, img_h, hfov_deg):
        hfov = math.radians(hfov_deg)
        fx = (img_w * 0.5) / math.tan(hfov * 0.5)
        vfov = 2.0 * math.atan(math.tan(hfov * 0.5) * (img_h / img_w))
        fy = (img_h * 0.5) / math.tan(vfov * 0.5)
        cx = (img_w - 1) * 0.5
        cy = (img_h - 1) * 0.5
        return fx, fy, cx, cy

    @staticmethod
    def _dilate(mask: torch.Tensor, k: int):
        return F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)

    @staticmethod
    def _erode(mask: torch.Tensor, k: int):
        return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=k // 2)

    @classmethod
    def _close(cls, mask: torch.Tensor, k: int):
        return cls._erode(cls._dilate(mask, k), k)

    def _not_white_mask(self, bev255: torch.Tensor):
        white = torch.tensor([255.0, 255.0, 255.0], device=bev255.device, dtype=bev255.dtype).view(1, 3, 1, 1)
        return ((bev255 - white) ** 2).sum(dim=1) > (self.white_thr ** 2)

    def _mask_by_color_bank(self, bev255: torch.Tensor, color_bank: torch.Tensor, tol: float, exclude_white=True):
        colors = color_bank.to(device=bev255.device, dtype=bev255.dtype).view(1, -1, 3, 1, 1)
        x = bev255.unsqueeze(1)
        ok = ((x >= (colors - tol)) & (x <= (colors + tol))).all(dim=2)
        if exclude_white:
            not_white = self._not_white_mask(bev255).unsqueeze(1)
            ok = ok & not_white
        return ok

    def parse_bev_torch(self, bev_rgb: torch.Tensor):
        B, _, H, W = bev_rgb.shape
        device = bev_rgb.device
        dtype = bev_rgb.dtype

        bev255 = bev_rgb * 255.0

        bld_match = self._mask_by_color_bank(
            bev255, self.bld_rgb_bank, tol=self.tol_bld, exclude_white=True
        )
        bld_h = self.bld_height_bank.to(device=device, dtype=dtype).view(1, -1, 1, 1)
        height_candidates = bld_match.to(dtype) * bld_h
        height_map = height_candidates.max(dim=1).values
        bld_any = bld_match.any(dim=1)

        road_match = self._mask_by_color_bank(
            bev255, self.road_rgb_bank, tol=self.tol_road, exclude_white=True
        )
        road_any = road_match.any(dim=1)

        bld_any = self._close(bld_any.unsqueeze(1).float(), k=3)[:, 0] > 0.5
        road_any = self._close(road_any.unsqueeze(1).float(), k=3)[:, 0] > 0.5

        cx, cy = W // 2, H // 2
        r = int(self.cam_clear_r)
        x0 = max(0, cx - r)
        x1 = min(W, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(H, cy + r + 1)

        height_map[:, y0:y1, x0:x1] = 0.0
        bld_any[:, y0:y1, x0:x1] = False

        return height_map, bld_any, road_any

    def render_view_from_bev_torch(self, height_map_m: torch.Tensor, road_mask_bev: torch.Tensor):
        device = height_map_m.device
        dtype = height_map_m.dtype

        B, bev_h, bev_w = height_map_m.shape
        out_h = out_w = self.out_size

        ppm = bev_w / float(self.bev_meters)
        bev_cx = (bev_w - 1) * 0.5
        bev_cy = (bev_h - 1) * 0.5

        fx, fy, cx, cy = self.build_camera_intrinsics(out_w, out_h, self.hfov)
        pitch = math.radians(self.pitch)

        u = torch.arange(out_w, device=device, dtype=dtype).view(1, 1, out_w).expand(1, out_h, out_w)
        v = torch.arange(out_h, device=device, dtype=dtype).view(1, out_h, 1).expand(1, out_h, out_w)

        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        z_cam = torch.ones_like(x_cam)

        Xd = x_cam
        Yd = -y_cam
        Zd = z_cam

        if abs(pitch) > 1e-8:
            cos_p = math.cos(pitch)
            sin_p = math.sin(pitch)
            Yd2 = Yd * cos_p - Zd * sin_p
            Zd2 = Yd * sin_p + Zd * cos_p
            Yd, Zd = Yd2, Zd2

        norm = torch.sqrt(Xd * Xd + Yd * Yd + Zd * Zd + 1e-8)
        Xd = Xd / norm
        Yd = Yd / norm
        Zd = Zd / norm

        Xd = Xd.expand(B, -1, -1)
        Yd = Yd.expand(B, -1, -1)
        Zd = Zd.expand(B, -1, -1)

        depth = torch.zeros((B, out_h, out_w), device=device, dtype=dtype)
        hit_bld = torch.zeros((B, out_h, out_w), device=device, dtype=torch.bool)
        hit_ground = torch.zeros((B, out_h, out_w), device=device, dtype=torch.bool)
        alive = torch.ones((B, out_h, out_w), device=device, dtype=torch.bool)

        ts = torch.arange(self.step, self.t_far + self.step, self.step, device=device, dtype=dtype)

        height_flat = height_map_m.view(B, -1)
        road_flat = road_mask_bev.view(B, -1)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, out_h, out_w)

        for t in ts:
            if not alive.any():
                break

            X = t * Xd
            Y = self.cam_h + t * Yd
            Z = t * Zd

            col = bev_cx + X * ppm
            row = bev_cy - Z * ppm

            inb = (col >= 0) & (col <= (bev_w - 1)) & (row >= 0) & (row <= (bev_h - 1))
            valid = alive & inb
            if not valid.any():
                continue

            c = torch.clamp(torch.round(col).long(), 0, bev_w - 1)
            r = torch.clamp(torch.round(row).long(), 0, bev_h - 1)
            lin = r * bev_w + c
            Hb = height_flat[batch_idx, lin]

            hit = valid & (Hb > 0.5) & (Y >= 0.0) & (Y <= Hb)
            if hit.any():
                depth = torch.where(hit, t.expand_as(depth), depth)
                hit_bld = hit_bld | hit
                alive = alive & (~hit)

            below = valid & (Y < 0.0)
            if below.any():
                t_ground = (-self.cam_h) / (Yd + 1e-8)
                depth = torch.where(below, t_ground, depth)
                hit_ground = hit_ground | below
                alive = alive & (~below)

        depth = torch.where(depth <= 0, torch.full_like(depth, float(self.t_far)), depth)

        hit_road = torch.zeros((B, out_h, out_w), device=device, dtype=torch.bool)
        gh = hit_ground
        if gh.any():
            t = depth
            X = t * Xd
            Z = t * Zd
            col = bev_cx + X * ppm
            row = bev_cy - Z * ppm
            c = torch.clamp(torch.round(col).long(), 0, bev_w - 1)
            r = torch.clamp(torch.round(row).long(), 0, bev_h - 1)
            lin = r * bev_w + c
            road_sample = road_flat[batch_idx, lin]
            hit_road = gh & road_sample

        return depth, hit_bld, hit_road, hit_ground

    def forward(self, bev_hint: torch.Tensor) -> torch.Tensor:
        """
        Return geometry-aware feature map instead of RGB hint.
        """
        if bev_hint.ndim != 4 or bev_hint.shape[1] != 3:
            raise ValueError(f"Expected bev_hint [B,3,H,W], got {tuple(bev_hint.shape)}")

        bev_hint = bev_hint.float()

        height_map_m, bld_any, road_any = self.parse_bev_torch(bev_hint)
        road_mask_bev = road_any & (~bld_any)

        depth_m, hit_bld, hit_road, hit_ground = self.render_view_from_bev_torch(
            height_map_m=height_map_m,
            road_mask_bev=road_mask_bev,
        )

        depth_norm = 1.0 - torch.clamp(depth_m / float(self.t_far), 0.0, 1.0)

        geom_feat = torch.stack([
            depth_norm,
            hit_bld.float(),
            hit_road.float(),
            hit_ground.float(),
        ], dim=1)  # [B,4,H,W]

        out_feat = self.feature_head(geom_feat)  # [B,F,H,W]
        return out_feat
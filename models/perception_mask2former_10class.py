# -*- coding: utf-8 -*-
import os, csv
import numpy as np
from PIL import Image, ImageFont
import cv2
import torch

# 使用 HF 模型
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# ========== 路径与模型 ==========
LOCAL_MODEL_DIR = r"/home/yuebing/GenAI_density_urbanization/models/mask2former-swin-small-cityscapes-semantic"
device          = "cuda" if torch.cuda.is_available() else "cpu"

# 输入与输出
input_folder  = r"/home/yuebing/GenAI_density_urbanization/urban_data/reconstruction_sample/generated/label_single_level_cue/beautiful"
output_folder = r"/home/yuebing/GenAI_density_urbanization/urban_data/reconstruction_sample/generated/label_single_level_cue/seg/beautiful_viz"
hint_dir      = r"/home/yuebing/GenAI_density_urbanization/urban_data/reconstruction_sample/generated/label_single_level_cue/seg/beautiful"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(hint_dir, exist_ok=True)

report_file = os.path.join(output_folder, "segment_label_single_level_cue_10classes.csv")

# ========== 后处理参数 ==========
MORPH_KERNEL = (3, 3)
MORPH_OPEN_ITERS  = 1
MORPH_CLOSE_ITERS = 2
MIN_AREA          = 200

USE_DENSE_CRF     = False
CRF_ITERS         = 5
USE_GUIDED_FILTER = False
GF_RADIUS         = 6
GF_EPS            = 1e-6

SAVE_OVERLAY_VIZ  = True
SAVE_BW_VIZ       = False  # 这里暂时没用黑白图，如需要可以再加


def postprocess_binary(mask01, orig_img=None):
    """对 0/1 掩膜做形态学+去小连通域抛光"""
    mask = (mask01.astype(np.uint8) * 255)

    kernel = np.ones(MORPH_KERNEL, np.uint8)
    if MORPH_OPEN_ITERS > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITERS)
    if MORPH_CLOSE_ITERS > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITERS)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype('uint8'), 8)
    clean = np.zeros_like(mask, dtype='uint8')
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            clean[labels == i] = 255
    mask = clean

    return (mask > 0).astype(np.uint8)


# ========== HF 推理函数 ==========
def run_hf_mask2former(img_path, processor, model):
    """返回 seg_map：(H, W)，值域 0~18"""
    image = Image.open(img_path).convert("RGB")

    # 只做预处理，不让 processor 自动转 tensor
    encoded = processor(
        images=[image],
        return_tensors=None,
        padding=True
    )

    pixel_values_np = np.stack(encoded["pixel_values"], axis=0)  # (1, C, H, W)
    pixel_mask_np   = np.stack(encoded["pixel_mask"],   axis=0)  # (1, H, W)

    pixel_values = torch.tensor(pixel_values_np, dtype=torch.float32, device=device)
    pixel_mask   = torch.tensor(pixel_mask_np,   dtype=torch.bool,   device=device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    seg_maps = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]]  # (H, W)
    )
    seg_map = seg_maps[0]

    return seg_map.cpu().numpy().astype(np.int32)


def main():
    print("Loading local HuggingFace Mask2Former model...")
    processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(LOCAL_MODEL_DIR).to(device)
    model.eval()
    print("✔ Model loaded.\n")

    # Cityscapes 类别固定 19 类，顺序和 HF 模型保持一致
    CITYSCAPES_CLASSES = [
        "road", "sidewalk", "building", "wall", "fence",
        "pole", "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck",
        "bus", "train", "motorcycle", "bicycle"
    ]

    # ========= 你的 10 类 Target Classes =========
    TARGET_CLASSES = {
        1: {"name": "Road",       "keywords": ["road"]},
        2: {"name": "Sidewalk",   "keywords": ["sidewalk"]},
        3: {"name": "Building",   "keywords": ["building"]},
        4: {"name": "Wall_Fence", "keywords": ["wall", "fence"]},
        5: {"name": "Pole_Sign",  "keywords": ["pole", "traffic light", "traffic sign"]},
        6: {"name": "Vegetation", "keywords": ["vegetation", "tree"]},
        7: {"name": "Terrain",    "keywords": ["terrain"]},
        8: {"name": "Sky",        "keywords": ["sky"]},
        9: {"name": "Person",     "keywords": ["person", "rider"]},
        10:{"name": "Vehicle",    "keywords": ["car", "truck", "bus", "train", "motorcycle", "bicycle"]},
    }

    # ========= 根据 Cityscapes 类名自动收集对应原始 id =========
    group2ids = {}
    for gid, info in TARGET_CLASSES.items():
        kws = [kw.lower() for kw in info["keywords"]]
        ids = set()
        for i, cname in enumerate(CITYSCAPES_CLASSES):
            cname_l = cname.lower()
            if any(kw in cname_l for kw in kws):
                ids.add(i)
        group2ids[gid] = ids

    print("=== Target Class → Cityscapes IDs ===")
    for gid, info in TARGET_CLASSES.items():
        mapped_names = [CITYSCAPES_CLASSES[i] for i in sorted(group2ids[gid])]
        print(f"{gid:2d} {info['name']}: {mapped_names}")

    # 已处理集合
    done = set()
    if os.path.exists(report_file):
        with open(report_file, newline="", encoding="utf-8-sig") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if row:
                    done.add(row[0])

    new_file = not os.path.exists(report_file) or os.stat(report_file).st_size == 0
    fcsv = open(report_file, "a", newline="", encoding="utf-8-sig")
    wr = csv.writer(fcsv)
    if new_file:
        wr.writerow(["filename", "label_id", "label_name", "pixel_count", "pixel_ratio"])

    # ========= 遍历所有图片 =========
    for root, dirs, files in os.walk(input_folder):
        # 计算相对路径（用来复刻文件夹结构）
        rel_dir = os.path.relpath(root, input_folder)

        # 在输出目录中创建对应子文件夹
        out_subdir = os.path.join(output_folder, rel_dir)
        hint_subdir = os.path.join(hint_dir, rel_dir)

        os.makedirs(out_subdir, exist_ok=True)
        os.makedirs(hint_subdir, exist_ok=True)

        for filename in files:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            # if filename in done:
            #     print(f"跳过已处理: {filename}")
            #     continue

            img_path = os.path.join(root, filename)
            print(f"Processing {filename} ...")

            try:
                seg_map = run_hf_mask2former(img_path, processor, model)
            except Exception as e:
                print(f"  ⚠ 推理失败：{e}")
                continue

            # 合法类别范围过滤
            valid = np.where((seg_map >= 0) & (seg_map < len(CITYSCAPES_CLASSES)), seg_map, -1)

            orig = Image.open(img_path).convert("RGB")
            H, W = valid.shape

            # ========= 构建最终 multi_mask：0=background, 1..10=目标类 =========
            multi_mask = np.zeros((H, W), dtype=np.uint8)

            # 按 gid 顺序叠加（后写优先级高，如需前后景优先可调顺序）
            for gid in range(1, 11):
                src_ids = list(group2ids.get(gid, []))
                if not src_ids:
                    continue

                bin_mask = np.isin(valid, src_ids).astype(np.uint8)
                bin_mask = postprocess_binary(bin_mask, orig_img=orig)
                multi_mask[bin_mask == 1] = gid

            # ========= 像素统计 =========
            total = H * W
            for gid, info in TARGET_CLASSES.items():
                cls_mask = (multi_mask == gid)
                px = int(cls_mask.sum())
                wr.writerow([filename, gid, info["name"], px, px / total])

            bg_mask = (multi_mask == 0)
            bg_px = int(bg_mask.sum())
            wr.writerow([filename, 0, "background", bg_px, bg_px / total])

            # ========= 保存训练用掩膜（0~10 的 label）=========
            save_name = os.path.splitext(filename)[0] + ".png"
            Image.fromarray(multi_mask.astype(np.uint8)).save(
                os.path.join(hint_subdir, save_name)
            )

            # ========= 可视化（彩色语义图）=========
            if SAVE_OVERLAY_VIZ:
                color_palette = {
                    0:(0,0,0),            # background

                    1:(128,64,128),       # Road
                    2:(244,35,232),       # Sidewalk
                    3:(70,70,70),         # Building
                    4:(190,153,153),      # Wall_Fence
                    5:(153,153,153),      # Pole_Sign

                    6:(107,142,35),       # Vegetation
                    7:(152,251,152),      # Terrain
                    8:(135,206,235),      # Sky
                    9:(220,20,60),        # Person
                    10:(0,0,142),         # Vehicle
                }

                viz = np.zeros((H, W, 3), dtype=np.uint8)
                for cid, color in color_palette.items():
                    viz[multi_mask == cid] = color

                Image.fromarray(viz).save(
                    os.path.join(out_subdir, f"{os.path.splitext(filename)[0]}_viz.png")
                )


    fcsv.close()
    print("\n✔ 完成")
    print("多类掩膜保存到：", hint_dir)
    print("CSV 保存到：", report_file)
    print("可视化保存到：", output_folder)


if __name__ == "__main__":
    try:
        _ = ImageFont.truetype("simsun.ttf", size=14)
    except:
        pass
    main()

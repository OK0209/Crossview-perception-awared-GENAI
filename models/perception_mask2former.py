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
input_folder  = r"/home/yuebing/GenAI_density_urbanization/urban_data/images_new"
output_folder = r"/home/yuebing/GenAI_density_urbanization/urban_data/mask2seg_mask_road_building_sky_tree_viz"
hint_dir      = r"/home/yuebing/GenAI_density_urbanization/urban_data/mask2seg_mask_road_building_sky_tree"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(hint_dir, exist_ok=True)

report_file = os.path.join(output_folder, "segment_report.csv")

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
SAVE_BW_VIZ       = True


def postprocess_binary(mask01, orig_img=None):
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

    # 1）让 processor 只做预处理，不做 tensor 转换
    encoded = processor(
        images=[image],          # 注意这里仍然用 list
        return_tensors=None,     # 关键：不要让它自动转成 BatchFeature/tensor
        padding=True             # 随便填一下，单张图也没事
    )

    # encoded 现在是一个字典：
    # encoded["pixel_values"] : 长度为1的 list，每个元素是 (C, H, W) 的 np.ndarray
    # encoded["pixel_mask"]   : 长度为1的 list，每个元素是 (H, W) 的 np.ndarray

    pixel_values_np = np.stack(encoded["pixel_values"], axis=0)  # (1, C, H, W)
    pixel_mask_np   = np.stack(encoded["pixel_mask"],   axis=0)  # (1, H, W)

    # 2）我们自己转成 torch.Tensor 并丢到 GPU/CPU
    pixel_values = torch.tensor(pixel_values_np, dtype=torch.float32, device=device)
    # pixel_mask 一般是 0/1，我们转成 bool
    pixel_mask   = torch.tensor(pixel_mask_np,   dtype=torch.bool,   device=device)

    # 3）前向推理
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    # 4）后处理成语义分割图
    seg_maps = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]]  # (H, W)
    )
    seg_map = seg_maps[0]  # 取 batch 里的第 0 张

    return seg_map.cpu().numpy().astype(np.int32)



def main():

    print("Loading local HuggingFace Mask2Former model...")
    processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(LOCAL_MODEL_DIR).to(device)
    model.eval()
    print("✔ Model loaded.\n")

    # Cityscapes 类别固定 19 类
    CITYSCAPES_CLASSES = [
        "road", "sidewalk", "building", "wall", "fence",
        "pole", "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck",
        "bus", "train", "motorcycle", "bicycle"
    ]

    ROAD_KEYWORDS      = {"road", "street", "highway"}
    BUILDING_KEYWORDS  = {"building", "structure"}
    SKY_KEYWORDS       = {"sky"}
    TREE_KEYWORDS      = {"vegetation", "tree"}

    road_ids = {i for i, name in enumerate(CITYSCAPES_CLASSES)
                if any(kw in name.lower() for kw in ROAD_KEYWORDS)}
    building_ids = {i for i, name in enumerate(CITYSCAPES_CLASSES)
                    if any(kw in name.lower() for kw in BUILDING_KEYWORDS)}
    sky_ids = {i for i, name in enumerate(CITYSCAPES_CLASSES)
               if any(kw in name.lower() for kw in SKY_KEYWORDS)}
    tree_ids = {i for i, name in enumerate(CITYSCAPES_CLASSES)
                if any(kw in name.lower() for kw in TREE_KEYWORDS)}

    print("[Info] 道路类：", [CITYSCAPES_CLASSES[i] for i in sorted(road_ids)])
    print("[Info] 建筑类：", [CITYSCAPES_CLASSES[i] for i in sorted(building_ids)])
    print("[Info] 天空类：", [CITYSCAPES_CLASSES[i] for i in sorted(sky_ids)])
    print("[Info] 树木类：", [CITYSCAPES_CLASSES[i] for i in sorted(tree_ids)])

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

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        if filename in done:
            print(f"跳过已处理: {filename}")
            continue

        img_path = os.path.join(input_folder, filename)
        print(f"Processing {filename} ...")

        try:
            seg_map = run_hf_mask2former(img_path, processor, model)
        except Exception as e:
            print(f"  ⚠ 推理失败：{e}")
            continue

        valid = np.where((seg_map >= 0) & (seg_map < len(CITYSCAPES_CLASSES)), seg_map, -1)

        base_mask = np.zeros(valid.shape, dtype=np.uint8)
        base_mask[np.isin(valid, list(road_ids))]     = 1
        base_mask[np.isin(valid, list(building_ids))] = 2
        base_mask[np.isin(valid, list(sky_ids))]      = 3
        base_mask[np.isin(valid, list(tree_ids))]     = 4

        orig = Image.open(img_path).convert("RGB")

        road_bin     = postprocess_binary((base_mask == 1).astype(np.uint8), orig)
        building_bin = postprocess_binary((base_mask == 2).astype(np.uint8), orig)
        sky_bin      = postprocess_binary((base_mask == 3).astype(np.uint8), orig)
        tree_bin     = postprocess_binary((base_mask == 4).astype(np.uint8), orig)

        multi_mask = np.zeros(valid.shape, dtype=np.uint8)
        multi_mask[sky_bin == 1]      = 3
        multi_mask[building_bin == 1] = 2
        multi_mask[tree_bin == 1]     = 4
        multi_mask[road_bin == 1]     = 1

        H, W = multi_mask.shape
        total = H * W
        wr.writerow([filename, 1, "road",     int((multi_mask == 1).sum()), int((multi_mask == 1).sum())/total])
        wr.writerow([filename, 2, "building", int((multi_mask == 2).sum()), int((multi_mask == 2).sum())/total])
        wr.writerow([filename, 3, "sky",      int((multi_mask == 3).sum()), int((multi_mask == 3).sum())/total])
        wr.writerow([filename, 4, "tree",     int((multi_mask == 4).sum()), int((multi_mask == 4).sum())/total])
        wr.writerow([filename, 0, "other",    int((multi_mask == 0).sum()), int((multi_mask == 0).sum())/total])

        save_name = os.path.splitext(filename)[0] + ".png"
        Image.fromarray(multi_mask.astype(np.uint8)).save(
            os.path.join(hint_dir, save_name)
        )

        if SAVE_OVERLAY_VIZ:
            palette = {
                0: (0, 0, 0),
                1: (255, 0, 0),
                2: (255, 255, 255),
                3: (135, 206, 235),
                4: (34, 139, 34),
            }
            viz = np.zeros((H, W, 3), dtype=np.uint8)
            for cid, color in palette.items():
                viz[multi_mask == cid] = color

            Image.fromarray(viz).save(
                os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_viz.png")
            )

    fcsv.close()
    print("\n✔ 完成")
    print("掩膜保存到：", hint_dir)
    print("CSV 保存到：", report_file)
    print("可视化保存到：", output_folder)


if __name__ == "__main__":
    try:
        _ = ImageFont.truetype("simsun.ttf", size=14)
    except:
        pass
    main()

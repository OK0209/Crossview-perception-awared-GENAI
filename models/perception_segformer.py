# -*- coding: utf-8 -*-
import os, csv
import numpy as np
from PIL import Image, ImageFont
import cv2

from mmseg.apis import init_model, inference_model

# ========== 路径与模型 ==========
config_file     = r"/home/yuebing/GenAI_density_urbanization/models/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py"
checkpoint_file = r"/home/yuebing/GenAI_density_urbanization/models/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth"
device          = "cuda:0"  # 无GPU时改为 "cpu"

# 输入与输出

input_folder  = r"/home/yuebing/GenAI_density_urbanization/urban_data/images_new"          # 原始图像
output_folder = r"/home/yuebing/GenAI_density_urbanization/urban_data/seg_mask_road_building_sky_tree_viz" # 日志/可视化
hint_dir      = r"/home/yuebing/GenAI_density_urbanization/urban_data/seg_mask_road_building_sky_tree" # 二值掩膜（训练/下游）
os.makedirs(output_folder, exist_ok=True)
os.makedirs(hint_dir, exist_ok=True)

report_file = os.path.join(output_folder, "segment_report_b5.csv")  # 统计（road vs non-road）

# ========== 可选：后处理参数 ==========
MORPH_KERNEL = (3, 3)   # 形态学核
MORPH_OPEN_ITERS  = 1
MORPH_CLOSE_ITERS = 2
MIN_AREA          = 200  # 小连通域剔除阈值（按你图像分辨率调整）

USE_DENSE_CRF     = False  # 如需CRF边界收紧，改为 True（需安装 pydensecrf）
CRF_ITERS         = 5

USE_GUIDED_FILTER = False  # 若不用CRF，可改用导向滤波（需 opencv-contrib-python）
GF_RADIUS         = 6
GF_EPS            = 1e-6

SAVE_OVERLAY_VIZ  = True   # 可视化（原图叠加）用于人眼检查
SAVE_BW_VIZ       = True   # 纯黑白可视化（不影响训练）


def postprocess_binary(mask01, orig_img=None):
    """ 对 0/1 掩膜做抛光：开闭运算、去小连通域、（可选）CRF/导向滤波 """
    mask = (mask01.astype(np.uint8) * 255)

    # 形态学开/闭，去毛刺与小洞
    kernel = np.ones(MORPH_KERNEL, np.uint8)
    if MORPH_OPEN_ITERS > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITERS)
    if MORPH_CLOSE_ITERS > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITERS)

    # 去掉极小连通域
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype('uint8'), 8)
    clean = np.zeros_like(mask, dtype='uint8')
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            clean[labels == i] = 255
    mask = clean

    # 可选：CRF（利用原图边界）
    if USE_DENSE_CRF and orig_img is not None:
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
            H, W = mask.shape
            n_labels = 2
            labels01 = (mask > 0).astype(np.uint8)
            U = unary_from_labels(labels01, n_labels, gt_prob=0.7)
            U = np.ascontiguousarray(U)
            d = dcrf.DenseCRF2D(W, H, n_labels)
            d.setUnaryEnergy(U)
            pair = create_pairwise_bilateral(sdims=(10, 10), schan=(5, 5, 5),
                                             img=np.asarray(orig_img), chdim=2)
            d.addPairwiseEnergy(pair, compat=5)
            Q = d.inference(CRF_ITERS)
            refined = np.argmax(np.array(Q), axis=0).reshape(H, W).astype('uint8')
            mask = (refined * 255).astype('uint8')
        except Exception as e:
            print(f"  ⚠ DenseCRF 失败，跳过：{e}")

    # 可选：导向滤波（轻量边界收紧）
    if USE_GUIDED_FILTER and orig_img is not None:
        try:
            if len(orig_img.shape) == 3 and orig_img.shape[2] == 3:
                guide = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2GRAY)
            else:
                guide = np.array(orig_img)
            mask_f = (mask / 255.0).astype(np.float32)
            mask_f = cv2.ximgproc.guidedFilter(guide.astype(np.float32), mask_f, GF_RADIUS, GF_EPS)
            mask = (mask_f > 0.5).astype('uint8') * 255
        except Exception as e:
            print(f"  ⚠ 导向滤波失败，跳过：{e}")

    return (mask > 0).astype(np.uint8)  # 返回 0/1


def main():
    # 初始化模型
    model = init_model(config_file, checkpoint_file, device=device)

    # 取类别&调色盘
    if not hasattr(model, "dataset_meta"):
        raise RuntimeError("模型没有 dataset_meta，无法获取 classes/palette。")
    classes = model.dataset_meta.get("classes", None)
    palette = model.dataset_meta.get("palette", None)
    if classes is None or palette is None:
        raise RuntimeError("dataset_meta 缺少 classes 或 palette。")

    # 定义“道路”类别关键字并自动收集其 id
    ROAD_KEYWORDS = {"road", "street", "highway"}
    BUILDING_KEYWORDS = {"building", "structure"}
    SKY_KEYWORDS = {"sky"}
    TREE_KEYWORDS = {"tree", "vegetation"}  # Cityscapes 里 vegetation 也可算树木

    road_ids = {i for i, name in enumerate(classes)
                if any(kw in name.lower() for kw in ROAD_KEYWORDS)}

    building_ids = {i for i, name in enumerate(classes)
                    if any(kw in name.lower() for kw in BUILDING_KEYWORDS)}

    sky_ids = {i for i, name in enumerate(classes)
               if any(kw in name.lower() for kw in SKY_KEYWORDS)}

    tree_ids = {i for i, name in enumerate(classes)
                if any(kw in name.lower() for kw in TREE_KEYWORDS)}

    print("[Info] 道路类：", [classes[i] for i in sorted(road_ids)])
    print("[Info] 建筑类：", [classes[i] for i in sorted(building_ids)])
    print("[Info] 天空类：", [classes[i] for i in sorted(sky_ids)])
    print("[Info] 树木类：", [classes[i] for i in sorted(tree_ids)])


    # 已处理集合（根据 CSV）
    done = set()
    if os.path.exists(report_file):
        with open(report_file, newline="", encoding="utf-8-sig") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if row:
                    done.add(row[0])

    # 打开CSV（road/non-road二行统计）
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

        # 验证图像
        try:
            Image.open(img_path).verify()
        except Exception as e:
            print(f"  ⚠ PIL 无法打开，跳过：{e}")
            continue

        # 推理
        try:
            data_sample = inference_model(model, img_path)
        except Exception as e:
            print(f"  ⚠ 推理失败，跳过：{e}")
            continue

        # 取分割图
        if not hasattr(data_sample, "pred_sem_seg"):
            print("  ⚠ 无 pred_sem_seg，跳过")
            continue
        seg = data_sample.pred_sem_seg
        seg_tensor = seg.data if hasattr(seg, "data") else seg
        seg_map = seg_tensor.squeeze().cpu().numpy().astype(np.int32)

        # 生成 5 类掩膜：0=其他, 1=道路, 2=建筑, 3=天空, 4=树木
        valid = np.where((seg_map >= 0) & (seg_map < len(classes)), seg_map, -1)

        base_mask = np.full(valid.shape, 0, dtype=np.uint8)  # 初始化为 "其他"
        base_mask[np.isin(valid, list(road_ids))] = 1      # 道路
        base_mask[np.isin(valid, list(building_ids))] = 2  # 建筑
        base_mask[np.isin(valid, list(sky_ids))] = 3       # 天空
        base_mask[np.isin(valid, list(tree_ids))] = 4      # 树木

        # 读取原图（供后处理/可视化）
        orig = Image.open(img_path).convert("RGB")

        # 各类二值掩膜
        road_bin     = (base_mask == 1).astype(np.uint8)
        building_bin = (base_mask == 2).astype(np.uint8)
        sky_bin      = (base_mask == 3).astype(np.uint8)
        tree_bin     = (base_mask == 4).astype(np.uint8)

        # 分别后处理（可选）
        road_bin     = postprocess_binary(road_bin, orig_img=orig)
        building_bin = postprocess_binary(building_bin, orig_img=orig)
        sky_bin      = postprocess_binary(sky_bin, orig_img=orig)
        tree_bin     = postprocess_binary(tree_bin, orig_img=orig)

        # 重新合成回 multi_mask
        # 可以按“从后景到前景”的顺序叠加：天空→建筑→树木→道路
        multi_mask = np.full(valid.shape, 0, dtype=np.uint8)
        multi_mask[sky_bin == 1]      = 3
        multi_mask[building_bin == 1] = 2
        multi_mask[tree_bin == 1]     = 4
        multi_mask[road_bin == 1]     = 1
        # 统计（以全部像素为分母；如要排除 ignore 区，改成 (valid>=0).sum()）
        H, W = multi_mask.shape
        total = H * W

        road_px     = int((multi_mask == 1).sum())
        building_px = int((multi_mask == 2).sum())
        sky_px      = int((multi_mask == 3).sum())
        tree_px     = int((multi_mask == 4).sum())
        other_px    = int((multi_mask == 0).sum())

        wr.writerow([filename, 1, "road",     road_px,     road_px / total])
        wr.writerow([filename, 2, "building", building_px, building_px / total])
        wr.writerow([filename, 3, "sky",      sky_px,      sky_px / total])
        wr.writerow([filename, 4, "tree",     tree_px,     tree_px / total])
        wr.writerow([filename, 0, "other",    other_px,    other_px / total])

        # === 保存训练用掩膜：不缩放，严格 0/1 ===
        save_name = os.path.splitext(filename)[0] + ".png"
        Image.fromarray(multi_mask.astype(np.uint8)).save(
            os.path.join(hint_dir, save_name)
        )
        if SAVE_OVERLAY_VIZ:
            color_palette = {
                0: (0, 0, 0),        # 黑：其他
                1: (255,0, 0),      # 绿：道路
                2: (255, 255, 255),  # 白/灰：建筑
                3: (135, 206, 235),  # 浅蓝：天空
                4: (34, 139, 34),    # 深绿：树木
            }
            H, W = multi_mask.shape
            viz_img = np.zeros((H, W, 3), dtype=np.uint8)
            for cls_id, color in color_palette.items():
                viz_img[multi_mask == cls_id] = color
            Image.fromarray(viz_img).save(
                os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_viz.png")
            )

    fcsv.close()
    print("\n✔ 完成。二分类掩膜已保存到：", hint_dir)
    print("  报表：", report_file)
    if SAVE_BW_VIZ or SAVE_OVERLAY_VIZ:
        print("  可视化预览：", output_folder)


if __name__ == "__main__":
    # 可选：字体加载（目前未直接用到）
    try:
        _ = ImageFont.truetype("simsun.ttf", size=14)
    except:
        pass
    main()

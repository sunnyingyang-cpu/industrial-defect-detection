#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_neu_det.py — 将 NEU-DET 数据集标注转换为 YOLOv5 训练格式。

NEU-DET 是面向工业表面质量检测的经典公开数据集 (热轧钢带 6 类缺陷)，
本脚本把不同来源的标注统一成 YOLOv5 所需的 `<class_id> <x_c> <y_c> <w> <h>`(归一化) 格式，
并按比例切分 train/val/test，产出可直接被 `data/defect_neu.yaml` 引用的目录结构。

支持两种常见发布版目录结构 (脚本会自动递归查找，无需手动展平):
  A) 扁平:  IMAGES/1.jpg + ANNOTATIONS/1.xml   (同名同 stem)
  B) 分类:  images/crazing/crazing_1.jpg + annotations/crazing/crazing_1.xml

标注输入格式: 自动识别 (优先级 VOC-XML > 原生 txt)
  voc  每个图片同名 .xml，PASCAL VOC 格式：<object><name><bndbox>xmin,ymin,xmax,ymax</bndbox></object>
  neu  每个图片同名 .txt，东北大学原生格式：每行 `class_name x1 y1 x2 y2`(像素, 左上+右下)

用法示例:
  # 最省事：不传 --format，脚本按文件自动判断 (推荐)
  python tools/prepare_neu_det.py \
      --src datasets/raw/IMAGES \
      --ann datasets/raw/ANNOTATIONS \
      --out datasets/NEU-DET

  # 也可显式指定
  python tools/prepare_neu_det.py \
      --src datasets/raw/images --ann datasets/raw/annotations \
      --out datasets/NEU-DET --format voc

输出结构:
  <out>/
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt

依赖: 仅标准库；读取图片尺寸优先用 Pillow，未安装时回退到 NEU-DET 固定尺寸 200×200。
"""
import argparse
import os
import glob
import shutil
import random
import xml.etree.ElementTree as ET

# 必须与 data/defect_neu.yaml 中的 names 顺序一致
CLASSES = [
    "crazing",          # 龟裂
    "inclusion",        # 夹杂
    "patches",          # 斑块
    "pitted_surface",   # 凹坑表面
    "rolled-in_scale",  # 轧入氧化皮
    "scratches",        # 划痕
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def read_image_size(path):
    """读取图片宽高；无 Pillow 时回退 NEU-DET 固定尺寸。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.width, im.height
    except Exception:
        return 200, 200  # NEU-DET 标准尺寸


def parse_voc(ann_path):
    boxes = []
    tree = ET.parse(ann_path)
    root = tree.getroot()
    for obj in root.iter("object"):
        name = obj.findtext("name")
        if name is None:
            continue
        name = name.strip()
        if name not in CLASSES:
            continue
        cls_id = CLASSES.index(name)
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            x1 = float(bnd.findtext("xmin"))
            y1 = float(bnd.findtext("ymin"))
            x2 = float(bnd.findtext("xmax"))
            y2 = float(bnd.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def parse_neu(ann_path):
    boxes = []
    with open(ann_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            name = parts[0].strip()
            if name not in CLASSES:
                continue
            cls_id = CLASSES.index(name)
            try:
                x1, y1, x2, y2 = map(float, parts[1:5])
            except ValueError:
                continue
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def to_yolo_lines(boxes, img_w, img_h):
    lines = []
    for cls_id, x1, y1, x2, y2 in boxes:
        # 裁剪到图片边界 + 容错
        x1 = max(0.0, min(x1, img_w))
        y1 = max(0.0, min(y1, img_h))
        x2 = max(0.0, min(x2, img_w))
        y2 = max(0.0, min(y2, img_h))
        if x2 <= x1 or y2 <= y1:
            continue
        xc = ((x1 + x2) / 2.0) / img_w
        yc = ((y1 + y2) / 2.0) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def find_annotation(ann_root, stem):
    """在标注根目录下递归查找 <stem>.xml 或 <stem>.txt，返回 (路径, 格式)。"""
    for ext, fmt in ((".xml", "voc"), (".txt", "neu")):
        hits = glob.glob(os.path.join(ann_root, "**", stem + ext), recursive=True)
        if hits:
            return hits[0], fmt
    return None, None


def main():
    ap = argparse.ArgumentParser(description="NEU-DET 标注转 YOLOv5 格式")
    ap.add_argument("--src", required=True, help="原始图片目录 (可含分类子目录，脚本递归查找)")
    ap.add_argument("--ann", required=True, help="标注目录 (与图片同名，可含分类子目录，递归查找)")
    ap.add_argument("--out", required=True, help="输出根目录")
    ap.add_argument("--format", choices=["voc", "neu"], default=None,
                    help="标注格式；不传则按文件扩展名自动识别 (推荐)")
    ap.add_argument("--split", default="0.8,0.1,0.1",
                    help="train/val/test 比例, 默认 0.8,0.1,0.1")
    ap.add_argument("--seed", type=int, default=42, help="随机种子(切分可复现)")
    args = ap.parse_args()

    ratios = [float(x) for x in args.split.split(",")]
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6, "split 必须归一化为 3 个数且和为 1"

    # 递归收集图片 (兼容扁平 / 分类子目录两种结构)
    img_files = []
    for ext in IMG_EXTS:
        img_files.extend(glob.glob(os.path.join(args.src, "**", "*" + ext), recursive=True))
        img_files.extend(glob.glob(os.path.join(args.src, "**", "*" + ext.upper()), recursive=True))
    img_files = sorted(set(img_files))
    if not img_files:
        raise SystemExit(f"在 {args.src} 未找到图片")

    random.seed(args.seed)
    random.shuffle(img_files)

    n = len(img_files)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    splits = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * (n - n_train - n_val)
    )

    counters = {"train": 0, "val": 0, "test": 0, "skip": 0}
    fmt_seen = set()
    for img_path, split in zip(img_files, splits):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        ann_path, fmt = find_annotation(args.ann, stem)
        if ann_path is None:
            counters["skip"] += 1
            continue
        if args.format is not None and fmt != args.format:
            counters["skip"] += 1
            continue
        fmt_seen.add(fmt)

        parser = parse_voc if fmt == "voc" else parse_neu
        img_w, img_h = read_image_size(img_path)
        boxes = parser(ann_path)
        yolo = to_yolo_lines(boxes, img_w, img_h)
        if not yolo:
            counters["skip"] += 1
            continue

        img_dst = os.path.join(args.out, "images", split)
        lbl_dst = os.path.join(args.out, "labels", split)
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        shutil.copy(img_path, os.path.join(img_dst, os.path.basename(img_path)))
        with open(os.path.join(lbl_dst, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(yolo) + "\n")
        counters[split] += 1

    print("转换完成:")
    print(f"  识别到的标注格式: {', '.join(sorted(fmt_seen)) or '无'}")
    print(f"  train={counters['train']}  val={counters['val']}  test={counters['test']}  skip={counters['skip']}")
    print(f"  输出目录: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

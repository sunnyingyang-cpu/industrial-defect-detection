#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_detect_compare.py

对比：原始 NEU-DET 图 vs CLAHE 增强图，用同一 YOLOv5 best.pt 做推理，
直观展示传统图像增强是否能帮助深度学习模型召回低对比度缺陷。

输出：docs/figures/preprocessing/detect_compare_{class}.png
"""
import os
import subprocess
import shutil
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMG = os.path.join(ROOT, "datasets", "NEU-DET", "images", "test")
WORK = os.path.join(ROOT, "runs", "detect_compare")
OUT = os.path.join(ROOT, "docs", "figures", "preprocessing")
WEIGHTS = os.path.join(ROOT, "runs", "train", "exp4", "weights", "best.pt")
PY = r"D:\Miniconda3\envs\industrial\python.exe"

# 选择模型已有一定响应、且 CLAHE 可能带来变化的样本：
#   - crazing: 在 conf 0.25 下几乎没有检出，选 0.10 阈值下有一框的 crazing_189
#   - rolled-in_scale: 在 conf 0.25 下已有低置信度检出，选 rolled-in_scale_197
SAMPLES = [
    ("crazing", "crazing_189.jpg"),
    ("rolled-in_scale", "rolled-in_scale_197.jpg"),
]


def prepare_dirs():
    # 沙箱内无法直接删除文件，这里只保证目录存在，同名文件会被后续写入覆盖
    for d in ["src_original", "src_clahe", "out_original", "out_clahe"]:
        os.makedirs(os.path.join(WORK, d), exist_ok=True)


def apply_clahe(src, dst):
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    cv2.imwrite(dst, clahe)


def run_detect(source, save_project, save_name):
    cmd = [
        PY, "detect.py",
        "--weights", WEIGHTS,
        "--source", source,
        "--img-size", "640",
        "--conf-thres", "0.10",
        "--iou-thres", "0.45",
        "--device", "0",
        "--save-txt",
        "--save-conf",
        "--project", save_project,
        "--name", save_name,
        "--exist-ok",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_labels(path):
    """返回 (框数量, 平均置信度, 最高置信度)。"""
    if not os.path.exists(path):
        return 0, 0.0, 0.0
    confs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    confs.append(float(parts[5]))
                except ValueError:
                    continue
    if not confs:
        return 0, 0.0, 0.0
    return len(confs), float(np.mean(confs)), float(np.max(confs))


def draw_compare(cls, fname):
    stem = os.path.splitext(fname)[0]
    orig_img_path = os.path.join(WORK, "out_original", fname)
    clahe_img_path = os.path.join(WORK, "out_clahe", fname)
    orig_label_path = os.path.join(WORK, "out_original", "labels", f"{stem}.txt")
    clahe_label_path = os.path.join(WORK, "out_clahe", "labels", f"{stem}.txt")

    orig_img = cv2.imread(orig_img_path)
    clahe_img = cv2.imread(clahe_img_path)
    if orig_img is None or clahe_img is None:
        print(f"[warn] missing inference output for {cls}")
        return
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)

    n_orig, mean_orig, max_orig = parse_labels(orig_label_path)
    n_clahe, mean_clahe, max_clahe = parse_labels(clahe_label_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original input\n{n_orig} boxes, mean conf={mean_orig:.2f}, max conf={max_orig:.2f}", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(clahe_img)
    axes[1].set_title(f"CLAHE-enhanced input\n{n_clahe} boxes, mean conf={mean_clahe:.2f}, max conf={max_clahe:.2f}", fontsize=11)
    axes[1].axis("off")
    fig.suptitle(f"Detection comparison — {cls}", fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT, f"detect_compare_{cls}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[ok] {cls}: orig={n_orig} boxes, clahe={n_clahe} boxes -> {os.path.basename(out)}")


def main():
    prepare_dirs()
    for cls, fname in SAMPLES:
        src = os.path.join(TEST_IMG, fname)
        shutil.copy(src, os.path.join(WORK, "src_original", fname))
        apply_clahe(src, os.path.join(WORK, "src_clahe", fname))
    print("[ok] prepared original & CLAHE source images")

    run_detect(os.path.join(WORK, "src_original"), WORK, "out_original")
    run_detect(os.path.join(WORK, "src_clahe"), WORK, "out_clahe")

    for cls, fname in SAMPLES:
        draw_compare(cls, fname)
    print("done.")


if __name__ == "__main__":
    main()

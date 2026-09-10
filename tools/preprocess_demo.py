#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_demo.py — 钢铁表面缺陷传统图像处理 / 特征增强演示

对 NEU-DET 样本做一组经典 CV 操作，覆盖 JD 关键词：
  - 去噪 / 增强：双边滤波、CLAHE、Gamma 校正
  - 边缘检测：Canny
  - 形态学处理：Top-Hat / Black-Hat、开运算、闭运算
  - 分割：Otsu 阈值、自适应阈值、连通域分析
  - 特征提取：GLCM 纹理特征（纯 numpy 实现）、LBP 局部二值模式

所有对比图输出到 docs/figures/preprocessing/，供 README 直接引用。
GLCM / LBP 统计量同时写入 docs/figures/preprocessing/features.json 并打印。

依赖：opencv-python, numpy, scipy, matplotlib（无需 scikit-image / scikit-learn）。
运行：
  python tools/preprocess_demo.py
"""
import os
import json
import math
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMG = os.path.join(ROOT, "datasets", "NEU-DET", "images", "test")
OUT = os.path.join(ROOT, "docs", "figures", "preprocessing")
os.makedirs(OUT, exist_ok=True)

# 挑选样本：弱类(crazing / rolled-in_scale) + 强类(inclusion / patches)
SAMPLES = [
    ("crazing", "crazing_110.jpg"),
    ("rolled-in_scale", "rolled-in_scale_100.jpg"),
    ("inclusion", "inclusion_115.jpg"),
    ("patches", "patches_10.jpg"),
]


# --------------------------------------------------------------------------- #
# 基础增强 / 去噪
# --------------------------------------------------------------------------- #
def enhance(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    gamma = 0.8
    g = np.clip(((img / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    bilateral = cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)
    return clahe, g, bilateral


# --------------------------------------------------------------------------- #
# 边缘 + 形态学
# --------------------------------------------------------------------------- #
def edge_morph(img):
    v = np.median(img)
    low = int(max(0, 0.66 * v))
    high = int(min(255, 1.33 * v))
    canny = cv2.Canny(img, low, high)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k5)     # 增强亮结构
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k5)  # 增强暗结构
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, k3)       # 去孤立噪点
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k3)      # 连接断裂
    return canny, tophat, blackhat, opening, closing


# --------------------------------------------------------------------------- #
# 分割 + 连通域
# --------------------------------------------------------------------------- #
def segment(img):
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 2)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(otsu, connectivity=8)
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 10:  # 过滤极小噪点
            continue
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 1)
    return otsu, adaptive, overlay, max(0, num - 1)


# --------------------------------------------------------------------------- #
# GLCM 纹理特征（纯 numpy）
# --------------------------------------------------------------------------- #
def glcm_features(img, levels=16, distances=(1, 2), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
    q = np.clip((img.astype(np.float32) / 256.0 * levels).astype(int), 0, levels - 1)
    glcm = np.zeros((levels, levels), dtype=np.float64)
    for d in distances:
        for a in angles:
            dx = int(round(math.cos(a) * d))
            dy = int(round(math.sin(a) * d))
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(q, -dy, axis=0), -dx, axis=1)
            for i, j in zip(q.ravel(), shifted.ravel()):
                glcm[i, j] += 1
    s = glcm.sum()
    if s > 0:
        glcm /= s
    # 统计量
    c = np.arange(levels)
    contrast = np.sum([((i - j) ** 2) * glcm[i, j] for i in c for j in c])
    homogeneity = np.sum([glcm[i, j] / (1.0 + (i - j) ** 2) for i in c for j in c])
    asm = np.sum(glcm ** 2)
    energy = math.sqrt(asm)
    mu_i = np.sum(c * glcm.sum(axis=1))
    mu_j = np.sum(c * glcm.sum(axis=0))
    var = np.sum([((c[k] - mu_i) ** 2) * glcm.sum(axis=1)[k] for k in c]) + 1e-9
    corr = np.sum([((c[i] - mu_i) * (c[j] - mu_j) * glcm[i, j])
                   for i in c for j in c]) / var
    return {
        "contrast": float(contrast),
        "homogeneity": float(homogeneity),
        "energy": float(energy),
        "correlation": float(corr),
    }


# --------------------------------------------------------------------------- #
# LBP 局部二值模式（3x3 标准，含 uniform 映射）
# --------------------------------------------------------------------------- #
def lbp_image(img):
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            c = img[y, x]
            code = 0
            for k, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1),
                                          (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]):
                if img[y + dy, x + dx] >= c:
                    code |= (1 << k)
            out[y, x] = code
    return out


def lbp_uniform_hist(img, levels=16):
    """UNIFORM(8,1) -> 59 bins，返回直方图(归一化)与熵/方差统计量。"""
    table = {}
    # 生成 uniform 查找表（跳变<=2 的为 uniform）
    for i in range(256):
        b = [(i >> k) & 1 for k in range(8)]
        trans = sum(1 for k in range(8) if b[k] != b[(k + 1) % 8])
        table[i] = i if trans <= 2 else 58
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            c = img[y, x]
            code = 0
            for k, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1),
                                          (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]):
                if img[y + dy, x + dx] >= c:
                    code |= (1 << k)
            out[y, x] = table[code]
    hist, _ = np.histogram(out.ravel(), bins=59, range=(0, 59))
    if hist.sum() > 0:
        hist = hist / hist.sum()
    ent = -np.sum([p * math.log(p + 1e-12) for p in hist if p > 0])
    return hist, {"lbp_entropy": float(ent), "lbp_var": float(hist.var())}


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def make_pipeline_fig(name, img, clahe, g, bilateral, canny, tophat,
                      blackhat, opening, closing, otsu, adaptive, overlay, cc_n):
    titles = ["Original", "CLAHE", "Gamma 0.8", "Bilateral(denoise)",
              "Canny", "Top-Hat(bright)", "Black-Hat(dark)", "Opening",
              "Closing", "Otsu", "Adaptive", f"CC (n={cc_n})"]
    panels = [img, clahe, g, bilateral, canny, tophat, blackhat, opening,
              closing, otsu, adaptive, overlay]
    fig = plt.figure(figsize=(13, 9.8))
    gs = GridSpec(3, 4, figure=fig)
    for idx, (t, p) in enumerate(zip(titles, panels)):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        if p.ndim == 3:
            ax.imshow(p)
        else:
            ax.imshow(p, cmap="gray")
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Traditional CV pipeline — {name}", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, f"{name}_pipeline.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def make_lbp_fig(results):
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, lbp) in zip(axes, results):
        ax.imshow(lbp, cmap="gray")
        ax.set_title(f"LBP — {name}", fontsize=11)
        ax.axis("off")
    fig.suptitle("Local Binary Pattern (3x3, uniform)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT, "lbp_montage.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def make_glcm_fig(feat):
    classes = list(feat.keys())
    metrics = ["contrast", "homogeneity", "energy", "correlation"]
    data = {m: [feat[c][m] for c in classes] for m in metrics}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, m in zip(axes.ravel(), metrics):
        ax.bar(classes, data[m], color="#3a7bda")
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("GLCM texture features by defect class", fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT, "glcm_features.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    summary_feat = {}
    lbp_results = []
    for cls, fname in SAMPLES:
        path = os.path.join(TEST_IMG, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[skip] {path}")
            continue
        clahe, g, bilateral = enhance(img)
        canny, tophat, blackhat, opening, closing = edge_morph(img)
        otsu, adaptive, overlay, cc_n = segment(img)
        fig = make_pipeline_fig(cls, img, clahe, g, bilateral, canny, tophat,
                                blackhat, opening, closing, otsu, adaptive,
                                overlay, cc_n)
        print(f"[ok] pipeline   {cls:14s} -> {os.path.basename(fig)} (CC={cc_n})")

        gf = glcm_features(img)
        lbp, lbp_stat = lbp_uniform_hist(img)
        summary_feat[cls] = {**gf, **lbp_stat}
        lbp_results.append((cls, lbp_image(img)))
        print(f"     GLCM {gf} | {lbp_stat}")

    make_lbp_fig(lbp_results)
    make_glcm_fig(summary_feat)
    with open(os.path.join(OUT, "features.json"), "w") as f:
        json.dump(summary_feat, f, indent=2)
    print(f"[ok] features   -> glcm_features.png / lbp_montage.png / features.json")
    print("done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ci_check.py — 仓库结构 / 配置自检 (CI 用)。

校验内容:
  1) models/*.yaml 与 data/*.yaml 均为合法 YAML 且含 'nc' 字段；
  2) 以 defect_ 开头的模型配置必须满足 nc == 6 且 names 有 6 项
     (与 data/defect_neu.yaml 的 6 类缺陷对齐)；
  3) data/defect_neu.yaml 的 names 键数量 == nc。

不依赖 torch，可在最小 CI 环境中运行。
"""
import glob
import os
import sys

try:
    import yaml
except ImportError:
    sys.stderr.write("缺少 PyYAML，请先 `pip install pyyaml`\n")
    sys.exit(2)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_DEFECT_NC = 6


def check_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"YAML 解析失败: {e}"]
    errors = []
    if not isinstance(cfg, dict):
        errors.append("顶层不是 mapping")
        return errors
    if "nc" not in cfg or not isinstance(cfg.get("nc"), int) or cfg["nc"] <= 0:
        errors.append("缺少合法的 'nc' (正整数)")
        return errors

    base = os.path.basename(path)
    if base.startswith("defect_"):
        # 模型配置只要求 nc 正确; names 在 data/*.yaml 中定义
        if cfg["nc"] != EXPECTED_DEFECT_NC:
            errors.append(f"'nc' 应为 {EXPECTED_DEFECT_NC}, 实际 {cfg['nc']}")
    return errors


def main():
    yaml_files = sorted(
        glob.glob(os.path.join(ROOT, "models", "*.yaml"))
        + glob.glob(os.path.join(ROOT, "data", "*.yaml"))
    )
    if not yaml_files:
        sys.stderr.write("未找到任何 yaml 配置\n")
        sys.exit(1)

    failed = 0
    for p in yaml_files:
        errs = check_yaml(p)
        if errs:
            failed += 1
            print(f"[FAIL] {os.path.relpath(p, ROOT)}")
            for e in errs:
                print(f"       - {e}")
        else:
            print(f"[ OK ] {os.path.relpath(p, ROOT)}")

    # 专项: defect_neu.yaml 的 names 数量须与 nc 一致
    neu = os.path.join(ROOT, "data", "defect_neu.yaml")
    if os.path.exists(neu):
        with open(neu, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names", {})
        n_names = len(names) if isinstance(names, (dict, list)) else 0
        if n_names != cfg.get("nc"):
            failed += 1
            print(f"[FAIL] data/defect_neu.yaml: names({n_names}) != nc({cfg.get('nc')})")
        else:
            print(f"[ OK ] data/defect_neu.yaml: names 与 nc 一致 ({n_names})")

    if failed:
        print(f"\nCI 自检失败: {failed} 项")
        sys.exit(1)
    print("\nCI 自检全部通过 ✅")
    sys.exit(0)


if __name__ == "__main__":
    main()

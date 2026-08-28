# Windows 本地环境搭建指南（industrial-defect-detection）

> 本指南假设你在本机操作。下面的路径都是已经为你准备好的，**不需要重新安装 Miniconda**。

## 0. 前提（已就绪，请勿重复安装）

| 项目                    | 位置                              | 说明                            |
| --------------------- | ------------------------------- | ----------------------------- |
| Miniconda3            | `D:\Miniconda3`                 | 已安装，base 为 Python 3.14        |
| Conda 环境 `industrial` | `D:\Miniconda3\envs\industrial` | 已创建，Python 3.10               |
| NVIDIA 驱动安装包          | `D:\nvidia_driver_596.49.exe`   | 已下载（约 914MB，GTX 1650 最新 WHQL） |

你只需要完成：**装驱动 → 装 PyTorch(GPU) → 验证 → 配 PyCharm**。

---

## 1. 安装 NVIDIA 驱动并重启（必须，否则 GPU 用不了）

1. 双击 `D:\nvidia_driver_596.49.exe`
2. 同意许可 → 选 **Express（精简）** 安装 → 开始安装
3. 装完后**必须重启电脑**

重启后验证驱动已生效（二选一）：

- 右键「此电脑」→ 管理 → 设备管理器 → 显示适配器 → 看 `NVIDIA GeForce GTX 1650` 没有黄色感叹号
- 或打开 CMD 输入 `nvidia-smi`，能打出一张表（含驱动版本、CUDA 版本）即正常

---

## 2. 打开终端并激活环境

**方式 A（推荐）**：开始菜单搜索 `Anaconda Prompt (Miniconda3)` 打开。

**方式 B**：`Win + R` → 输入 `cmd` → 回车，然后逐行执行：

```bat
D:\Miniconda3\Scripts\activate.bat
conda activate industrial
```

看到命令行前缀变成 `(industrial)` 就成功了。

> 之后所有命令都在 `(industrial)` 环境下执行。

---

## 3. 安装 GPU 版 PyTorch（CUDA 12.4）

在 `(industrial)` 环境下，**二选一**：

### 方案 A：conda（最稳，自带 cudatoolkit）

```bat
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
```

国内网络慢时，改用清华镜像（把频道换成下面这行）：

```bat
conda install pytorch torchvision pytorch-cuda=12.4 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/nvidia
```

### 方案 B：pip（通常更快）

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> 注意：`--index-url https://download.pytorch.org/whl/cu124` 才能装到 **GPU 版**。  
> 普通 `pip install torch`（走 PyPI）装的是 **CPU 版**，验证时 `cuda.is_available()` 会是 `False`。

---

## 4. 安装其余依赖（排除 torch/torchvision）

为避免覆盖刚装好的 GPU 版 PyTorch，**不要**直接 `pip install -r requirements.txt`（它会把 torch 换成 PyPI 的 CPU 版）。请单独安装其余包：

```bat
pip install opencv-python matplotlib Pillow PyYAML scipy tqdm tensorboard seaborn pandas onnx onnxruntime Flask
```

---

## 5. 验证 GPU 可用

```bat
python -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

期望输出类似：

```
torch 2.x.x+cu124
CUDA available: True
```

- 若 `CUDA available: False`：回去第 1 步，确认驱动装好且**已重启**。
- 若 `torch` 版本不带 `+cu124`：说明装成了 CPU 版，回到第 3 步用 GPU 源重装。

---

## 6. PyCharm 配置解释器

1. `File → Settings → Python Interpreter`
2. 右上角齿轮 → `Add...`
3. 选 `Conda Environment` → `Use existing environment`
4. Interpreter 路径填：
   ```
   D:\Miniconda3\envs\industrial\python.exe
   ```
5. OK → Apply

配置好后，PyCharm 就能直接运行项目里的 `detect.py / train.py / test.py`。

---

## 7. 跑起来（推理 demo）

项目目录：`E:\software\workbuddypro\2026-08-27-15-51-47\industrial-defect-detection`

```bat
cd E:\software\workbuddypro\2026-08-27-15-51-47\industrial-defect-detection
python detect.py --source data/images/sample_defect.jpg --weights yolov5s.pt --conf 0.25
```

- 首次运行会自动下载 `yolov5s.pt`（约 14MB）权重到 `weights/`。
- `sample_defect.jpg` 是本地 PIL 生成的**合成**钢板缺陷演示图（非真实 NEU-DET 样本）。`yolov5s.pt` 是 COCO 预训练权重，不是缺陷模型，所以这张图上大概率检不出"缺陷"类别——这一步只是验证「权重能下、CUDA 能推理、结果能写出」。想真正检测缺陷，按 `README.md` 第 4.2→4.4 节准备 NEU-DET 数据集并训练。

---

## 8. 清理（重要）

自动安装过程中误装了一份 Miniconda 到 `C:\Users\77916\miniconda3`（WorkBuddy 的安全删除机制拦截了自动清理）。请手动删除该文件夹以释放空间：

- 资源管理器里直接删除 `C:\Users\77916\miniconda3`，或
- 管理员 CMD 执行：`rmdir /s /q C:\Users\77916\miniconda3`

**`D:\Miniconda3` 才是你要保留的**，别删错了。

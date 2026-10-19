# Regression of Used Car Prices

使用 PyTorch 多层感知机（MLP）预测二手车价格，基于 Kaggle 竞赛
[Playground Series - Season 4, Episode 9: Regression of Used Car Prices](https://www.kaggle.com/competitions/playground-series-s4e9)。

目标是预测二手车的价格（`price`），评估指标为 **RMSE（均方根误差）**。

## 项目结构

```
.
├── main.py              # 入口：校验数据 → 预处理 → 训练 → 生成提交文件
├── preprocess.py        # 数据预处理：标准化 + 独热编码
├── train.py             # 模型定义、训练循环、评估与预测
├── sha1_check.py        # 下载数据后的完整性校验（sha1）
├── gpu_monitor.py       # 训练时后台采样 nvidia-smi，绘制显存/利用率/功耗曲线
├── requirements.txt     # Python 依赖
├── .gitignore
└── used_car_prices/     # 数据目录（需自行下载，见下）
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

## 快速开始

### 1. 环境

- Python 3.10+
- NVIDIA GPU（可选，无 GPU 会自动退回 CPU 训练）
- 仅 `gpu_monitor.py` 需要系统装有 `nvidia-smi`（Windows 需在 PATH 中）

```bash
pip install -r requirements.txt
```

> GPU 版 PyTorch 请按官方指引安装：
> https://pytorch.org/get-started/locally/

### 2. 下载数据

从竞赛页面下载数据并解压到 `used_car_prices/` 目录，最终应包含：

- `train.csv`
- `test.csv`
- `sample_submission.csv`

### 3. 校验数据完整性

程序启动时会自动对 `train.csv` 和 `test.csv` 做 sha1 校验，防止下载的文件损坏：

| 文件 | sha1 |
|---|---|
| `used_car_prices/train.csv` | `b9ebb482a7f97ecaec17cb7fb347ac78ba1beca1` |
| `used_car_prices/test.csv` | `392a215375f587c4dcc297d2faf5535d90dcae1c` |

### 4. 运行

```bash
python main.py
```

运行结束会在 `used_car_prices/` 下生成 `submission.csv`，可直接提交到 Kaggle。

## 方法说明

- **特征工程**（`preprocess.py`）
  - 数值特征（如 `model_year`、`milage`）做标准化（z-score）；
  - 类别特征（`brand`、`model`、`engine` 等）做独热编码；
  - 缺失值用均值（标准化后为 0）填充。

- **模型**（`train.py`）
  - 三层 MLP：`Linear(输入维度, 10) → ReLU → Linear(10, 10) → ReLU → Linear(10, 1)`；
  - 损失函数为 MSE，训练时对预测值取对数，以更贴合 RMSE 指标。

- **GPU 优化**
  - 训练数据一次性常驻显存，用 GPU 端 `randperm` 采样 batch，避免逐 batch 的 CPU→GPU 传输开销；
  - 评估阶段使用 `torch.no_grad()` 并分批计算，控制显存峰值。

## 注意事项

- **Windows 下必须「先 import torch 再 import pandas/numpy」**（`main.py` 已按此顺序处理），否则可能出现
  `OSError: [WinError 1114]` 的 DLL 初始化错误。若彻底解决，建议将 numpy 升级到 2.x 并将 pandas 升级到 2.2.2+。
- `gpu_monitor.py` 依赖 NVIDIA 驱动与 `nvidia-smi` 命令，无 NVIDIA 显卡时请勿启用。

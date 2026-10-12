# DIP-LiteDenoising

[English](README.md) | [中文](README.zh-CN.md)

## 最终实验结果

**[查看全部已完成实验](results/README.md)** · [实验目录](results/experiments.csv) · [完整最终指标](results/results.csv) · [框架对比图](results/figure/architecture_comparison.png)

核心模型命名为 **TMASR-Net（Temporal-Mean-Anchored Sub-pixel Residual Network；G4）**。它将 DeepInterpolation 的连续序列自监督思想、全分辨率时间均值与紧凑亚像素残差修正整合，两条预测分支均排除中心帧，无需干净参考图像进行训练。匹配 w30（60 帧上下文）时，G4 为 **139,204** 参数，原版架构 DI 为 **15,704,769** 参数：参数减少 **112.8 倍**，卷积算术减少 **80.4–83.9 倍**，RTX 5070 Ti 上批量 1 独立前向提速 **3.7–4.2 倍**。三记录 seed 0 的验证重建 PSNR 与 MSE-DI 的差均小于 **0.009 dB**。

公开结果包含 **124 个实验条目和 16,677 个标量指标**，覆盖容量、G1–G5 归因、原版 DI 的 MAE/MSE、结构与窗口、下采样加宽、仿真、标准局部 SSIM、效率、ROI/Suite2p、因果输入和新增记录验证。已有完整单种子结果与多种子汇总分别标识；各实验说明改动、测试目的和最终数值。贡献与测量定义见[解释报告](docs/technical_report.zh-CN.md)。模型前向和完整流程时间分列。

[事件 ROC 更正说明](docs/event_roc_correction.md)：已修复原始帧号到预测行号的映射，使用原有权重重新评价，并补充实测假阳性率。G4 两个仿真噪声条件的 AUC 均为 1.000000；各方法的灵敏度均达到 100%，正文以幅值和峰时指标区分动态响应。

![原版 DI 与 TMASR-Net（G4）的信息流](results/figure/architecture_comparison.png)

此图为解释性框架图，由 OpenAI imagegen 于 2026-10-12 生成并修正；接口没有暴露图像模型版本。图中文字、通道、尺寸及信息路径已对照实现核查，实验数值来自保存的测量结果。

DIP-LiteDenoising 是一个紧凑的 PyTorch 项目，用于实现和研究
DeepInterpolation 式自监督时间预测中的 U-Net 模型族。模型从目标中心帧两侧的邻近帧
预测该中心帧，输入中明确排除中心帧。本仓库适用于科学图像序列的架构检查、模型开发、
训练和推理。

同一个 Python 包还包含冻结的 `paper_evidence_v1` 实验流程，以及 405 项容量研究和后续
252 项目标函数—计算结构研究的轻量公开证据。仓库不包含下载的记录、预训练权重、
checkpoint、预测帧或可恢复图像的数组。

## 项目意义

高通量科学成像会持续产生长时间序列，模型容量、显存和设备执行效率可能成为数据处理的
限制因素。本项目将论文中考察的网络宽度、深度和卷积单元整理为明确、可调用的 PyTorch
模型族，并将冻结协议、可执行实验流程和公开审计证据放在同一个仓库内。

源码聚焦于三个可复用部分：

- 排除中心帧的时间窗口构造；
- 使用标准卷积、深度可分离卷积、倒残差和 Ghost 单元的可配置二维 U-Net；
- 最小训练、断点、清单记录和有序推理链路。

## 模型族

`bf` 表示基础特征宽度，`d` 表示编码器深度。

| 模型标签 | 基础宽度 | 深度 | 计算单元 |
|---|---:|---:|---|
| `bf4_d2_standard` | 4 | 2 | 标准双卷积 |
| `bf4_d3_standard` | 4 | 3 | 标准双卷积 |
| `bf8_d2_standard` | 8 | 2 | 标准双卷积 |
| `bf8_d3_standard_full_reference` | 8 | 3 | 标准双卷积 |
| `bf8_d4_standard` | 8 | 4 | 标准双卷积 |
| `bf16_d3_standard` | 16 | 3 | 标准双卷积 |
| `bf8_d3_depthwise` | 8 | 3 | 深度可分离卷积 |
| `bf8_d3_inverted_residual` | 8 | 3 | 倒残差单元 |
| `bf8_d3_ghost` | 8 | 3 | Ghost 单元 |

`compact_di` 是 `bf8_d3_standard_full_reference` 的易读别名。历史标签中的
`full_reference` 仅为模型注册名称，不表示训练使用了干净参考帧。

输入通道数由时间窗口决定。例如，
`WindowSpec(pre_frame=2, post_frame=2)` 会产生 4 个邻帧输入通道。中心目标帧始终
不进入输入。

## 项目结构

```text
DIP-LiteDenoising/
├── examples/
│   └── quickstart_synthetic.py
├── configs/
│   ├── paper_evidence_v1/  # 正式冻结的核心、机制与协议配置
│   ├── exploratory/        # 非确认性的探索筛选配置
│   └── splits/             # 公开拆分清单（不含像素）
├── docs/                   # 技术报告、复现和数据边界
├── evidence/
│   ├── capacity_study_v1/  # 405 项训练的轻量证据
│   └── paper_evidence_v1/  # 252 项训练的轻量证据
├── scripts/                # 流程、绘图和清单核验脚本
├── src/deepinterpolation_improve/
│   ├── config/       # 时间窗口、数据、训练和推理配置
│   ├── data/         # 时序数据集、窗口处理和合成数据
│   ├── inference/    # 保持帧顺序的 HDF5 推理
│   ├── manifest/     # 运行记录和输入兼容性检查
│   ├── models/       # U-Net 单元、注册表和论文模型族
│   ├── training/     # 损失函数、训练、断点及恢复
│   └── experiments/paper_evidence_v1/ # 集成的实验子包
├── tests/            # 核心功能与实验合同测试
├── LICENSE
├── NOTICE.md
├── THIRD_PARTY_NOTICES.md
├── README.md
├── README.zh-CN.md
└── pyproject.toml
```

## 环境要求

- Python 3.10 或更高版本
- PyTorch 2.8 或更高版本
- NumPy、h5py、tifffile、nibabel 和 tqdm

使用 CUDA 时，应先根据本机 CUDA 环境安装匹配的 PyTorch 版本。具体命令以
[PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)为准。

## 安装

Windows PowerShell：

```powershell
git clone https://github.com/Bohao-Ding/DIP-LiteDenoising.git
cd DIP-LiteDenoising
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Linux 或 macOS：

```bash
git clone https://github.com/Bohao-Ding/DIP-LiteDenoising.git
cd DIP-LiteDenoising
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## 论文证据实验流程

安装后，`dip-evidence` 与可复用核心使用同一个 Python 包。正式配置独立放在
`configs/paper_evidence_v1/`，探索筛选配置放在 `configs/exploratory/`。

不依赖本地记录即可检查源码与配置合同，并枚举冻结的核心矩阵：

```powershell
dip-evidence check
dip-evidence paper-core --no-path-check
```

完整本地复跑时，先创建被 Git 忽略的机器路径文件，填写三个 Neurofinder 记录根目录，
再核验记录与冻结拆分：

```powershell
Copy-Item configs/paths.example.json configs/paths.local.json
dip-evidence data-check --paths configs/paths.local.json
dip-evidence paper-core --paths configs/paths.local.json
dip-evidence paper-mechanism --paths configs/paths.local.json
```

后两条命令默认只输出任务计划。核对任务快照后再添加 `--execute`；继续已有本地
checkpoint 时添加 `--resume`。也可执行完整锁定流程：

```powershell
python scripts/run_paper_evidence_pipeline.py --paths configs/paths.local.json
```

无需训练或数据访问即可核验轻量公开证据：

```powershell
python scripts/build_distribution_manifest.py --check
```

公开路径派生文件为
`evidence/paper_evidence_v1/tables/source_mapping.public.csv`。其中机器绝对路径已替换为
仓库相对路径；该修订以及原私有证据清单的哈希均记录在
`evidence/distribution_manifest.json`。详见[复现说明](docs/reproduction.md)和
[完整中文技术报告](docs/technical_report.zh-CN.md)。

## 快速功能复现

列出全部模型标签：

```powershell
python examples/quickstart_synthetic.py --list-variants
```

使用 CPU 完成一次单轮训练和推理：

```powershell
python examples/quickstart_synthetic.py `
  --variant compact_di `
  --device cpu `
  --epochs 1 `
  --output-dir outputs/quickstart
```

在 Linux 或 macOS 中，请将 PowerShell 续行符替换为反斜杠，或将命令写在同一行。

该程序会在内存中生成小型钙成像风格序列，加入泊松—高斯噪声，训练指定模型，重新加载
`best.pt`，随后将有序预测写入 HDF5。程序不会下载数据，也不会计算 PSNR、SSIM、
推理延迟或论文表格。

预期运行产物如下：

```text
outputs/quickstart/
├── predictions.h5
└── training/
    ├── best.pt
    ├── checkpoint.pt
    ├── loss_history.csv
    └── manifest.json
```

这些文件已被 Git 忽略。

## 创建模型

使用论文中注册的模型变体：

```python
from deepinterpolation_improve.models import create_model_variant

model = create_model_variant(
    "bf8_d3_ghost",
    input_channels=4,
    output_channels=1,
)
```

使用 `compact_di` 创建紧凑型标准卷积配置：

```python
model = create_model_variant("compact_di", input_channels=4)
```

如果需要自定义宽度、深度或计算单元，可以使用底层 `create_model` 接口：

```python
from deepinterpolation_improve.models import create_model

model = create_model(
    "unet_single_1024",
    {
        "input_channels": 4,
        "output_channels": 1,
        "base_features": 12,
        "depth": 3,
        "block_type": "standard",
    },
)
```

## 使用 TIFF 序列训练

输入 TIFF 应保存为 `[时间, 高度, 宽度]` 序列。

```python
from pathlib import Path

from deepinterpolation_improve.config import TrainingConfig, WindowSpec
from deepinterpolation_improve.data import SingleTifDataset
from deepinterpolation_improve.models import create_model_variant
from deepinterpolation_improve.training import Trainer

movie_path = Path("path/to/movie.tif")
window = WindowSpec(pre_frame=2, post_frame=2, pre_post_omission=0)

train_data = SingleTifDataset(
    movie_path,
    window=window,
    start_frame=0,
    end_frame=799,
    randomize=True,
    seed=7,
)
validation_data = SingleTifDataset(
    movie_path,
    window=window,
    start_frame=804,
    end_frame=999,
    randomize=False,
)

model = create_model_variant("compact_di", train_data.input_shape[0])
config = TrainingConfig(
    output_dir=Path("outputs/my_run"),
    run_uid="my_run",
    model_name="compact_di",
    loss="mean_absolute_error",
    batch_size=4,
    epochs=75,
    learning_rate=1e-3,
    scheduler_step_size=25,
    scheduler_gamma=0.5,
    device="cuda",
    seed=7,
)

Trainer(model, train_data, validation_data, config).fit()
```

训练区间与验证区间应互不重叠，并根据所选时间窗口保留足够的时间净空。不同记录的长度
和采集条件不同，因此仓库不强制提供统一的数据划分。

## 恢复训练

使用相同的模型和数据配置，并将 `resume_from` 指向已有断点：

```python
config = TrainingConfig(
    output_dir=Path("outputs/my_run"),
    run_uid="my_run",
    model_name="compact_di",
    epochs=100,
    resume_from=Path("outputs/my_run/checkpoint.pt"),
)
```

断点会保存模型、优化器、学习率调度器、AMP 缩放器和训练器所需的随机状态。

## 推理

```python
from pathlib import Path

import torch

from deepinterpolation_improve.inference import InferenceRunner
from deepinterpolation_improve.models import create_model_variant

model = create_model_variant("compact_di", validation_data.input_shape[0])
checkpoint = torch.load(
    "outputs/my_run/best.pt",
    map_location="cpu",
    weights_only=False,
)
model.load_state_dict(checkpoint["model_state"])

InferenceRunner(
    model,
    validation_data,
    output_file=Path("outputs/my_run/predictions.h5"),
    batch_size=4,
    device="cpu",
    manifest_path=Path("outputs/my_run/manifest.json"),
).run()
```

推理默认使用数据集的有序副本，并同时保存预测结果和来源帧编号。清单校验会拒绝时间
窗口或输入尺寸不兼容的推理配置。

## 支持的数据接口

源码提供以下数据适配器：

- 内存中的 NumPy 数组；
- 单个或多个 TIFF 序列；
- 光学成像 HDF5 序列；
- 电生理二进制数据；
- Neurofinder 目录结构；
- NIfTI 格式 fMRI 体数据；
- 使用 JSON 索引的电影集合；
- 合成钙成像风格序列。

本仓库不分发原始记录。`evidence/` 仅包含仓库作者生成的聚合、运行级或帧级指标、元数据
与渲染图。使用外部数据集时，用户仍应自行检查其许可证和引用要求。

## 第三方数据与许可证边界

源码包含面向外部数据的读取接口和下载来源信息，其中包括 Neurofinder。这些接口不分发
数据，也不改变数据的权利状态。截至 2026 年 8 月 6 日，CodeNeuro 的公开
Neurofinder 仓库和 datasets 仓库中未注明明确的数据集许可证或数据集 DOI。公开下载
不应被解释为允许再次发表或再分发记录。使用者应从来源方获取第三方数据，并自行确认
拟议用途符合来源方条件。

Allen Institute 的 DeepInterpolation 项目受其自身许可证约束，其中包含 MIT License
未包含的限制。本仓库的 MIT License 只覆盖独立编写的 PyTorch 源码及其文档，不授予
Allen 源码、Neurofinder 数据或其他第三方材料的权利。具体来源与边界见
[NOTICE.md](NOTICE.md) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 测试

```powershell
python -m pytest -q
```

测试会在临时目录中创建数据和输出，覆盖可复用模型族、中心帧排除、数据顺序、断点恢复、
集成实验模型、严格 state-dict 加载、冻结任务矩阵、路径覆盖与证据分析合同。

## 适用边界

- 自监督训练中的中心目标帧是观测帧，不是干净真值图像。
- 成功预测中心帧不能单独证明噪声已经被去除、生物信号得到保持或下游分析得到改善。
- 公开证据可以审计两项冻结的本地研究，但仍不包含训练 checkpoint、原始记录和预测帧。
- 合成数据快速入口只验证程序能够运行；复现报告指标仍需外部记录并遵循冻结协议。
- 实际性能和显存占用取决于时间窗口、图像尺寸、批量大小、PyTorch 版本和硬件。
- `paper_evidence_v1` 的匹配评价片段曾在早期探索中被查看，不能表述为独立的选后测试集。

## 与 DeepInterpolation 的关系

本项目受到以下工作提出的自监督时间预测原理启发：

Lecoq, Jérôme, et al. “Removing Independent Noise in Systems Neuroscience Data
Using DeepInterpolation.” *Nature Methods*, vol. 18, 2021, pp. 1401–1408.
[https://doi.org/10.1038/s41592-021-01285-2](https://doi.org/10.1038/s41592-021-01285-2)。

Allen Institute 的原始项目位于
[AllenInstitute/deepinterpolation](https://github.com/AllenInstitute/deepinterpolation)。
本仓库不包含旧 TensorFlow 源码。

## 许可证

本仓库中独立编写的 PyTorch 源码采用 [MIT License](LICENSE)。该许可证不覆盖第三方
软件或数据。方法来源和复用边界见 [NOTICE.md](NOTICE.md) 与
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

当前未创建 Zenodo 归档或仓库 DOI。

## 本地研究与发布职责

本目录保留独立公开仓库历史。完整本地研究工作区由单独的 Git 仓库管理，并使用归并后的研究项目作为开发入口；不要将两个同名包安装到同一 Python 环境。本次保留 51 项公开证据的清单字节，未新增科学结果、未推送。维护上下文见 [MEMORY.md](MEMORY.md)。

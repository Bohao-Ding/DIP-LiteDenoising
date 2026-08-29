# 复现与审计说明

本说明区分“公开证据审计”和“从原始记录完整复跑”。前者不需要数据或 GPU；后者需要
使用者自行取得 Neurofinder 记录，并承担相应权利与计算资源责任。

## 1. 安装与软件自检

在仓库根目录创建隔离环境并安装可编辑包：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -q
dip-evidence check
```

测试使用临时目录，不需要 Neurofinder 数据。`dip-evidence check` 检查代码、配置与任务
矩阵，不开始训练。

## 2. 公开证据审计

```powershell
python scripts/build_distribution_manifest.py --check
```

该命令逐项核对公开文件的 SHA-256 与字节数，同时拒绝 checkpoint、可恢复图像类型、
机器绝对路径和超过 100 MiB 的单文件。清单顶层记录源协议、冻结哈希、原私有证据清单
哈希及公开修订规则。

关键入口：

- `evidence/capacity_study_v1/`：405 项容量研究；
- `evidence/paper_evidence_v1/paper_report.json`：252 项冻结汇总；
- `evidence/paper_evidence_v1/tables/`：配对效应与来源映射；
- `evidence/paper_evidence_v1/latency/`：设备协议汇总与原始轮次记录；
- `evidence/paper_evidence_v1/figures/`：冻结 Figure 6 的三种格式。

## 3. 本地路径配置

正式配置不保存个人机器路径。复制模板后只编辑被 `.gitignore` 排除的本地文件：

```powershell
Copy-Item configs/paths.example.json configs/paths.local.json
```

`recording_roots` 的三个值分别指向 `01.00`、`01.01`、`03.00` 的图像目录；
`split_pattern` 默认使用仓库内 `configs/splits/neurofinder/{recording}/{window}.json`。
不要将 `paths.local.json`、记录、checkpoint 或 `artifacts/` 提交到 Git。

验证数据根、帧形状与冻结拆分：

```powershell
dip-evidence data-check --paths configs/paths.local.json
```

验证失败时不要开始训练。路径文件的 SHA-256 会写入任务计划和运行记录，以便识别使用的
本地映射，但公开证据不会保存其绝对路径。

## 4. 任务计划与执行

先执行只读计划：

```powershell
dip-evidence paper-core --paths configs/paths.local.json
dip-evidence paper-mechanism --paths configs/paths.local.json
```

确认核心为 225 项、机制组为 27 项后，才添加 `--execute`。恢复中断任务时添加
`--resume`：

```powershell
dip-evidence paper-core --paths configs/paths.local.json --resume --execute
dip-evidence paper-mechanism --paths configs/paths.local.json --resume --execute
```

也可让锁定流程依次恢复训练、冻结、评价、延迟和报告：

```powershell
python scripts/run_paper_evidence_pipeline.py --paths configs/paths.local.json --device cuda
```

该流程会写入被 Git 忽略的 `artifacts/`。它不会覆盖仓库中的公开 `evidence/`。

## 5. 冻结与评价屏障

完整训练结束后，`paper-freeze` 只在 252 项均满足运行合同、且评价产物尚不存在时建立
冻结清单。其后评价入口会重新核对配置、拆分、loss history、摘要与 checkpoint 哈希。
任一冻结输入改变时应停止，而不是更新哈希绕过屏障。

```powershell
dip-evidence paper-freeze --paths configs/paths.local.json --execute
dip-evidence paper-evaluate --paths configs/paths.local.json --scope all --execute
dip-evidence paper-latency --paths configs/paths.local.json --device cuda --execute
dip-evidence paper-report --paths configs/paths.local.json --execute
```

延迟结论只适用于记录在输出中的设备、PyTorch/CUDA、FP16、channels-last、batch size、
预热与重复协议。

## 6. 可复现性边界

- 公开包可验证冻结表与图的完整性，但不能在没有外部记录和 checkpoint 时重新训练；
- Neurofinder 像素及可恢复数组被明确排除；
- 匹配评价片段曾在早期探索中查看，不是独立选后测试；
- 不应把 noisy-centre-frame fidelity 等同于 clean-reference 去噪或生物学有效性。

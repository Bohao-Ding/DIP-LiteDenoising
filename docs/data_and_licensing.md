# 数据、证据与许可证边界

## 仓库内材料

仓库包含独立编写的 PyTorch 源码、配置、测试、文档，以及作者生成的轻量数值证据、元数据
和图。`LICENSE` 适用于仓库中由项目作者独立创作且未另行声明的材料；它不改变任何第三方
软件、论文、方法或数据集的权利状态。

公开证据的逐文件哈希、大小、来源协议与修订规则见
`evidence/distribution_manifest.json`。其中不包含 checkpoint/state-dict、原始图像、预测帧
或可恢复图像的数组。

## Neurofinder

实验使用者必须从 CodeNeuro 来源自行取得 Neurofinder 记录：

- https://github.com/codeneuro/neurofinder
- https://github.com/codeneuro/datasets

截至 2026 年 8 月 6 日，本项目在上述仓库和已下载元数据中未识别到明确的数据集许可证
或数据集 DOI。公开下载、空许可证字段或可访问的对象存储链接均不等于公有领域状态，也不
自动授予再分发权。因此，本仓库只保留来源 URL、归档哈希、拆分哈希和不可恢复像素的统计
结果，不分发 Neurofinder TIFF、完整帧或等价数组。

## 公开路径映射的修订

冻结工作区的原始 `source_mapping.csv` 含机器本地绝对路径，未提交。公开派生文件命名为
`evidence/paper_evidence_v1/tables/source_mapping.public.csv`，将数据根与拆分路径替换为
仓库相对位置。原文件 SHA-256、公开文件路径和修订规则均记录在 distribution manifest 的
`redactions` 数组中；原私有证据清单本身不随包分发，但其真实 SHA-256 被保留。

## 使用者责任

使用者应自行确认其数据取得、处理、共享和发表是否符合来源方条件、所在机构政策及适用
法律。本文件是项目的技术与来源边界说明，不构成法律意见。引用方法或数据时，应分别引用
DeepInterpolation 论文、相应架构论文和 Neurofinder 来源，而不能只引用本仓库。

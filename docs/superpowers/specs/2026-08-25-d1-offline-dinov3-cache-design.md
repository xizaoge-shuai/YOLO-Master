# D1 Offline DINOv3 Feature Cache Design

## Goal

构建一个不在训练阶段加载 DINOv3 的目标检测路径：预先缓存冻结 DINOv3
多层特征，只优化 LatentMixture 和 Detect，并输出满足 8.24 与 P0 的证据。

## Architecture

缓存构建器以固定顺序读取数据集，对图像执行固定尺寸 letterbox 和 DINOv3
归一化。冻结的 DINOv3 提取 block 3、7、11 特征，并以 FP16 safetensors
保存。

训练数据集读取缓存特征和变换后的检测标签。三个 LatentMixture 分别生成
P3、P4、P5，随后连接 Ultralytics Detect。正式训练路径不构建 DINOv3，
也不保留原 YOLO backbone。

## Feature contract

默认输入尺寸为 640×640，DINOv3 patch size 为 16。缓存构建器必须从模型
配置读取 hidden size 和 patch size，不允许仅依赖预期值。

manifest 记录：

- schema_version
- dataset YAML 与 split
- 每个样本的稳定 sample_id、相对路径和图像 SHA256
- DINOv3 model ID 和 revision
- transformer block indices
- imgsz、letterbox、归一化参数及预处理 fingerprint
- dtype、shape、缓存文件 SHA256
- 标签变换需要的 ratio/pad 信息

预期 n-scale P3/P4/P5 输出通道为 64、128、256，空间尺寸为 80×80、40×40、
20×20。实测维度必须写入接口报告。

## Reproducibility

100 张准入子集由排序后的稳定 sample_id 取前 100 个生成。相同模型 revision、
数据集、图片内容和预处理配置重复构建时，manifest 内容及特征 checksum 必须
一致。

缓存文件通过临时文件写入，校验完成后原子改名。训练时禁止静默回退到在线
DINOv3。

## Training policy

optimizer 使用显式白名单，只允许 LatentMixture 和 Detect 参数。启动训练前
输出全部 trainable 参数名、参数量和模块分类；出现白名单外参数时立即失败。

缓存训练禁用 mosaic、随机裁剪、随机尺度等会破坏缓存空间对应关系的增强。
用于 P1 的 from-scratch 基线采用同一数据划分、输入尺寸及增强约束。

## Resource accounting

分别记录：

- cache build wall time 和 GPU-hours
- cache total bytes、bytes/sample 和顺序读取 MB/s
- cache build peak VRAM
- cached training peak VRAM 和 GPU-hours
- baseline training peak VRAM 和 GPU-hours

P1 同时报告 cold-start 成本和 warm-cache 成本，不隐藏缓存构建成本。

## Error handling

缺失缓存、checksum 不匹配、模型 revision 不同、预处理 fingerprint 不同、
shape 不匹配或标签样本不匹配时，训练必须在首个错误样本处失败，并输出
sample_id 和具体字段。

## Tests

- manifest schema 与路径稳定性单元测试
- 缓存写入、读取和 checksum round-trip 测试
- revision、预处理和图片 hash 不匹配测试
- 100 样本重复缓存确定性测试
- P3/P4/P5 和 Detect 接口维度测试
- trainable parameter 白名单测试
- checkpoint 不含 DINOv3 参数测试
- 单 batch CPU 测试和单 GPU 一轮 smoke
- train/val 端到端 P0 测试

## Acceptance

8.24 需要提交 100 张缓存 manifest、校验结果、磁盘/I/O/显存报告、维度表、
配置、复现命令和完整日志。

P0 需要在正式数据集上完成训练和验证，证明训练阶段未加载 DINOv3，且只有
LatentMixture 和 Detect 可训练。

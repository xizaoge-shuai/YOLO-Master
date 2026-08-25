# D1 8.24 准入检查状态

## 路线

冻结 DINOv3，离线缓存多层视觉特征，仅训练 LatentMixture 和 Detect。

## 当前已通过

- 环境安装：Python、PyTorch、CUDA、Transformers、Ultralytics 可用。
- 最小训练链路：DINOv3 + 3 个 LatentMixture 可完成一轮训练。
- 遥测：foundation loss、router KL、router module 数量可记录。
- checkpoint：不包含 DINOv3 教师参数。
- 复现配置：已有 args.yaml 和 results.csv。
- 目录治理：datasets_dir 与 runs_dir 已指向当前仓库。

## 当前在线 smoke 结果

- train/foundation_loss: 0.157331
- train/foundation_router_modules: 3
- train/foundation_router_kl: 0.0156593
- train/mixture_aux_loss: 3.0

该结果仅作为已有在线蒸馏链路证据，不作为正式 P0 结果。

## 尚未完成的 8.24 项目

- 固定 100 张图片的离线 DINOv3 多层特征缓存。
- 对同一输入重复构建缓存并验证 manifest/checksum 一致。
- 缓存总容量、单样本容量及顺序读取吞吐量。
- 缓存构建阶段和训练阶段的峰值 GPU 显存。
- DINOv3、LatentMixture、P3/P4/P5、Detect 的实测维度表。
- 离线缓存读取训练的一批次 forward/backward。
- trainable parameter 白名单审计。

## P0 通过条件

- 正式训练阶段不实例化、不加载 DINOv3。
- 训练参数仅包含 LatentMixture 和 Detect。
- 缓存与数据集、模型 revision、预处理配置强绑定。
- 完成训练和验证，保存 mAP、显存、耗时及完整日志。
- 提供配置、复现命令、结果文件和已知限制。

## P1/P2 后续

P1 使用至少两个数据集，与同预算从零训练检测器比较精度、峰值显存和
GPU-hours，并同时报告包含缓存构建的 cold-start 成本及复用缓存的
warm-cache 成本。

P2 扫描 LatentMixture 辅助损失权重和 DINOv3 模型规模。当前主分支已经
统一收集 LatentMixture 辅助损失，因此不能将该历史修复作为本课题新增成果。

<!-- CACHE100-EVIDENCE-BEGIN -->

## 100 图离线缓存实测结果

- 数据集：COCO128 train，按稳定 sample_id 排序取前 100 张。
- 模型：`Tooony133/dinov3-vits16-pretrain-lvd1689m`。
- revision：`fc6921f7a0b44d5b33ab4482cfed5443db6ccd81`。
- 输入尺寸：640×640。
- 缓存层：[3, 7, 11]。
- 缓存格式：FP16 safetensors。
- 缓存总量：368664800 bytes（351.59 MiB）。
- 单样本：3.52 MiB。
- 缓存构建耗时：10.5119 秒。
- 缓存构建 GPU-hours：0.002920。
- 峰值分配显存：84.22 MiB。
- 完整文件 warm-cache 顺序读取：4876.38 MiB/s。
- manifest SHA256：`6820a2f4ff724d5978e9b48567cf56e2400e6a91a195ba70ba7ae7638410d267`。
- 两次 manifest 一致性：PASS。
- 100 个特征文件一致性：PASS。
- DINOv3 三层实测形状：384×40×40。
- LatentMixture 规划输出：P3=64×80×80，P4=128×40×40，P5=256×20×20。

I/O 数值为完整读取全部缓存文件的 warm OS page-cache 测量，不表述为裸盘极限吞吐。

<!-- CACHE100-EVIDENCE-END -->

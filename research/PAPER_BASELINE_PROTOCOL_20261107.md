# 四篇论文思想适配与全量优先队列

用户要求直接按论文思想补检测基线，不下载作者源码；不做新增 GPU 冒烟、小子集训练或 pilot。已有训练检查点、源码和结果保留。本文是实验协议，不修改论文正文。

## 已有证据与本批缺口

FroFA、LOFF-TA 已有完整的 400/100 探索，各有不同强度及三种子结果，但没有本次 6471/548 全量端点。Transport-C1 覆盖了 Cache and Reuse 的未压缩特征几何增强近似，尚无量化缓存版本。没有 ViT-Adapter 空间先验加双向交互基线。YOLO-Master-N、YOLO11-N、YOLO12-N 已有全量 seed0 保存评估；YOLO13 当前没有实现或结果，不能用另一个网络改名填表。

## 四篇来源与固定适配

| 显示名 | 本地文件 / 原文位置 | 本项目实现 |
| --- | --- | --- |
| FroFA-C2 | Bär et al., CVPR 2024, §3；所附 PDF | 真实 C2 缓存；逐图/通道空间 min/max 归一化，独立亮度偏移 δ=0.1、clip、逆映射；零训练教师查询 |
| LOFF-TA-C2 | arXiv:2410.02527v1, §3.3 / Fig.2 | 真实 C2 翻转视角；连续特征缩放 [0.85,1.15]，通道空间标准差×0.05 的高斯噪声；标签同步；零训练教师查询 |
| Cache and Reuse-C1-INT8 | Yang et al., CVPRW 2024, §4.2–4.3 / Eq.1–3 / Alg.2 | 单原图缓存，逐图/层/通道分位数仿射量化，真实 UINT8 落盘；训练反量化后做翻转/连续缩放，标签同步；零教师查询及训练 RGB 读取 |
| ViT-Adapter-style (cache-side) | arXiv:2205.08534v4, §3 / Fig.2 / §4.4 | RGB 多尺度 SPM、双向 injector/extractor、原生稀疏采样注意力；三个交互对应缓存层3/7/11，64维、4头×4点、FFN比例0.25；输出残差加到相同 P3/P4/P5；q25主比较，另有q0对照 |

这是检测任务的思想适配。原论文的分类头、完整增强搜索、预训练骨干或优化配方不直接照搬。ViT-Adapter 原网络还会让注入 token 经过后续 ViT blocks，缓存侧版本不能等价实现这一步；本实现明确不称原架构复现。原论文已经讨论 SPM 补空间细节，并有 SPM-only 消融，不把 RGB 分支本身作为未经比较的新机制。

Cache and Reuse PDF 没给出分位数 k 的具体数值，本项目固定 k=0.001，标为适配选择。常量通道保护、分位数裁剪、真实字节编码与解码均保留。INT4 为单独存储消融，两个4bit码实际打包到一个字节。验证仍用共同未压缩 FP16 特征，压缩针对训练缓存。服务器为审计保留的原 FP16 文件不等于方法训练时实际消费的缓存量。

## 数据、训练与评估

- 完整 VisDrone2019-DET：6471 train / 548 development validation。100 epochs、batch8、640、DINOv3-S/16的3/7/11层、mean_value检测头、AdamW及既有余弦学习率/损失。
- 同种子共享初始检测头、图像顺序和请求翻转/连续缩放。FroFA 的有效几何为 C2 原尺度；LOFF-TA 使用真实翻转缓存；Cache and Reuse 只从 C1 特征翻转；这些是策略差异，均记录。
- q25：累计 floor(epoch×6471×0.25) 张真实教师图像，尾批计入；固定预算不等于固定墙钟时间。适配器及 Detail 在训练和验证都读取 RGB，其成本完整计入。
- 每10轮及最后一轮验证；主要端点固定第100轮的 pinned VisDrone参考 AP/AP50/AP75/AR500。内部诊断AP另列，不按best挑选主结果。
- 三个训练种子分别保存；均值/样本SD只使用独立种子。历史seed0成本复测不变成第四个种子。548图是开发验证集，不称独立测试集。
- GPU0串行，共享GPU资源；GPU1当前不可用。保留实际运行telemetry，观察计时不支持受控加速结论。

## 一次性全量优先队列：28项，保留原20项且去重

1. 四篇适配的 seed0：FroFA、LOFF-TA、Cache and Reuse INT8、ViT-Adapter-style q25。
2. Mix25/Online100 seed0成本复测；Blank0/Detail0/Adapter0；INT4 seed0存储消融。
3. 四篇主适配各补seed1/2。
4. 原五臂 Transport/Blank/Views/Detail/Lowpass 各补seed1/2。已开始的 Blank25-s1 在原源码/原job下从检查点继续，不重新从头跑。

原20项中重复的FroFA/LOFF-TA、Mix/Online及零查询配置只保留一次。旧监督进程在下一轮原子检查点及sidecar发布后切换；保存优化器、调度器、CPU/CUDA RNG、查询配额、源码、SHA及可能尾段开销。旧目录训练源码与冻结队列不改写；新增实现放在隔离worktree。主队列直接跑全量，独立case失败记录后继续其他case，不把失败写成完成或用小规模数字补位。

## 按用户截图组织五张表

| 表 | 行 | 指标与限制 |
| --- | --- | --- |
| Table1 完整检测器 | YOLO-Master-N / YOLO11-N / YOLO12-N / YOLO13 / 最终候选 | AP/AP50/AP75/AR500、训练设置、种子、来源；缺失用—。YOLO13未实现；Master旧续训成本不当干净100轮成本 |
| Table2 缓存与刷新 | C1 / C2 / Mix10/25/50 / Online100 | 最终参考AP、种子、查询图像/比例、cold/warm成本；历史结果与新成本复测分开 |
| Table3 相关方法适配 | FroFA / LOFF-TA / ViT-Adapter-style / Cache and Reuse | 最终参考AP、实际查询预算、RGB输入、额外参数、缓存量、方法适配说明；不填原分类论文的指标 |
| Table4 空间补偿消融 | Transport / Blank / Views / Detail / Lowpass，以及q0对照 | 固定预算的AP与配对种子差、结构/输入控制；保留全部失败与退化 |
| Table5 成本与资源 | 上述可比实验 | 缓存构建/压缩秒数、训练/验证/墙钟、cold charged/warm wall、显存、实际消费缓存字节、参数、教师调用、RGB与解码开销 |

cost中的解码和RGB读取已嵌套在墙钟内，不重复相加；归档历史构建时间属于计费估算，不伪装成本次独占GPU的实测。初始压缩缓存共享三种子，分别给单次训练cold与复用warm口径，不暗中摊薄。

四张图先保留用户要求的逻辑：方法总览；AP–预算Pareto（横轴同时给实际时间/查询预算，纵轴最终AP）；缓存误差及空间补偿可视化；精度–训练成本对比。误差条来自独立训练种子，N=1不画。跨数据集/骨干、严格等时间及正式test-dev仍是后续独立证据，不由本批代替。

## 实现与检查

`paper_adapter_model.py`、`paper_quantized_cache.py`、`paper_baseline_suite.py`及`paper_baseline_campaign.py`分担架构、压缩、训练接口与调度。复用已有完整训练/官方评分，不拉作者代码。启动前只做源码静态复核、语法和lint，不跑任何新GPU smoke、合成训练或小数据实验。阶段轻量指标自动commit并尝试push；同步状态单独记录，权重/缓存不入Git。

# 实验结果台账

整理日期：2026-11-06；最近核对：2026-11-07。本文只整理结果、协议与证据缺口，不修改论文或历史实验记录。

最近五臂表是空间细节候选的消融表，不能代替全项目主比较。完整结果还包括完整检测器、缓存与在线预算、FroFA/LOFF-TA 适配、重建与查询策略，以及冻结头机制诊断。不同轮次不合并均值或计算受控速度优势；重复归档不算新增实验，因此不报跨阶段总 run 数。

## 1. 对照方法为什么存在

| 方法或来源 | 对照作用 | 当前证据范围 |
| --- | --- | --- |
| YOLO-Master-N | 项目来源及完整检测器参照 | 完整 VisDrone 训练与参考评估；与冻结 DINO 系统网络不同 |
| YOLO11-N、YOLO12-N | 主流完整检测器参照 | 完整 VisDrone；配方和成本须单独披露 |
| C1、C2 | 单视角缓存与真实原图/翻转缓存 | pilot 与完整 VisDrone；C2 另有三种子和 VOC 子集 |
| Random Mix、Online100 | 同冻结骨干下的预算收益与全在线参考 | 完整 VisDrone；q0/q25/q100 三种子 |
| FroFA，CVPR 2024 | 检查直接冻结特征亮度增强是否足够 | 已完成检测思想适配 pilot；未完成全量同协议对照 |
| LOFF-TA，NeurIPS 2024 | 检查离线张量几何增强及噪声是否足够 | 已完成检测思想适配 pilot；不是原分类论文完整复现 |
| ViT-Adapter，ICLR 2023 | 检查空间先验与双向稀疏特征交互 | 尚无完成结果；新增缓存侧思想适配，不等同原网络 |
| Cache and Reuse，CVPRW 2024 | 检查单缓存增强及低比特存储 | Transport-C1 pilot覆盖未压缩近似；新增真实INT8/INT4缓存版本 |
| Transport、Rec/Correct | 区分几何传输与学习重建的作用 | 同预算 pilot、完整 VisDrone 单种子 |
| Blank、Views | 检查廉价填充和增加真实缓存视角 | 最近五臂中的强简单对照 |
| RGB Detail、Lowpass | 相同 4,672 参数分支，检验输入细节的增量 | 完整 VisDrone，固定 q25、seed0 |
| DiKE / REVIVE | 写作组织与机制干预的参考 | LLM 知识编辑论文，不能列为检测 baseline |

FroFA/LOFF-TA 的实现边界见[配置说明](../online_experiments/FEATURE_BASELINES.md)。写作范例为 [DiKE：ICLR 2026](../ccfa-workfiles/exemplars/disentangling-editing/cache/iclr-2026-disentangling-knowledge-representations-for-large-language-model-editing-paper-conference.md) 与 [REVIVE：ACL 2026](../ccfa-workfiles/exemplars/acl-1384/2026-acl-long-1384.md)。旧聊天中的 ORT 描述不能替代后者的身份核验。

## 2. 数据与评估口径

| 数据设置 | 训练 / 验证 | 阶段及限制 |
| --- | ---: | --- |
| COCO128 固定子集 | 80 / 20 | D1 pilot，非正式 COCO 训练集评估 |
| COCO2017 val1000 再划分 | 800 / 200 | 已有 cached latent 与 YOLO26n-scratch 的 seed0、100 epoch 日志；非 train2017 正式结果 |
| VisDrone500 | 400 / 100 | 特征增强、预算、查询策略等 pilot；内部 COCO-style 诊断 AP |
| 完整 VisDrone2019-DET | 6471 / 548 | 主训练与开发验证；没有 test-dev 结果 |
| VOC2007 trainval 子集 | 2000 / 400 | 从 trainval 抽样；内部诊断 AP，非官方 VOC2007 test AP |

以下 AP 均按 0–100 记录；COCO 旧日志的 0–1 mAP 另行标注。完整 VisDrone 主指标为固定最终 checkpoint 的 pinned-reference AP，另报 AP50/AP75/AR500；历史 best 是内部诊断指标选出的 checkpoint，不是搜索所有轮次后的最高参考 AP。

参考实现对应 [VisDrone 工具箱固定版本](https://github.com/VisDrone/VisDrone2018-DET-toolkit/tree/005445782213e20cb91bc50a597db3dd949e749a)。Python 端对全部 548 图评分，24 个合成案例及两张真实图与 Octave 数值对齐；不能声称全部数据在原生 MATLAB 中重新评分，也不能将内部 macro AP 混入主 AP。详见[参考评估说明](experiments/visdrone-reference-job-full-20260927/README.md)。

## 3. 完整 VisDrone：缓存与预算主线

100 epochs，冻结 DINOv3-S/16、640、mean_value；下表为最终参考 AP。“—”表示没有该种子结果。

| 方法 | seed0 | seed1 | seed2 | 均值 ± 样本 SD |
| --- | ---: | ---: | ---: | ---: |
| C1 | 9.1186 | — | — | — |
| C2 / q0 | 10.6568 | 10.1599 | 10.3375 | 10.3847 ± 0.2518 |
| Random Mix10 | 10.6708 | — | — | — |
| Random Mix25 | 11.3526 | 10.8883 | 10.8096 | 11.0168 ± 0.2934 |
| Random Mix50 | 11.2299 | — | — | — |
| Transport25 | 9.6350 | — | — | — |
| Rec / Correct25 | 9.7656 | — | — | — |
| Online100 | 11.3982 | 11.3985 | 11.6916 | 11.4961 ± 0.1693 |

C2、Mix25、Online100 的三种子配对差：Mix25−C2 为 +0.6321 ± 0.1396 AP；Online100−Mix25 为 +0.4792 ± 0.4191 AP。seed0 的 93.85% 在线收益恢复率不能概括全部种子；逐种子恢复率均值为 62.51% ± 29.67%。这些是少量种子的描述性统计。

证据：[seed0 原始参考结果](experiments/visdrone-reference-job-full-20260927/original/results.json)；[聊天原始日志](../ChatGPT_答辩PPT与稿子撰写_20261106-2309.md) L21577–21588、L22519–22534、L22682–22699。2026-11-07 已找回并归档[种子1/2的六个原始全量运行](experiments/full-visdrone-confirm-job-start-VIVUsfpt/README.md)，核对各100轮记录和共600个计划哈希。AP为保存的参考评估记录，未重新评分；历史评估缺checkpoint/预测哈希绑定，旧provenance含两个dirty文件，不能写成新重算或干净快照。原始路径已在归档JSON中归一化，数值未改。

## 4. 最近五臂：空间细节消融

固定提交 `2b13720c718dbd14f99d4e722671d930a261997e` 的五臂完整训练和结果归档均已核验；实验标识 `job-20261022-hjP4y3Lf`。历史 10 月 22 日“待启动”文档保留其当时含义，本节依据完成归档更新。

共同配置：完整 6471/548、100 epochs、batch8、seed0；连续缩放 [0.85,1.15]、翻转概率0.5、相同 mean_value 头初始化和配对计划；训练 q25，共 161775 teacher-image queries，包含尾批次。固定最终轮评分，未按 best 选择。

| 臂 | 操作 | AP | AP50 | AP75 | 观察训练秒数 |
| --- | --- | ---: | ---: | ---: | ---: |
| Transport25 | C2 空间传输 | 9.6350 | 20.9255 | 7.9773 | 38532.9205 |
| Blank25 | Transport＋真实空白图特征填充 | 9.9984 | 21.4894 | 8.3233 | 32277.3573 |
| Views25 | 3尺度×2翻转真实缓存＋残余传输/填充 | 10.0912 | 21.7731 | 8.2518 | 39386.0559 |
| Detail25 | Blank＋RGB P3 分支 | 12.3453 | 26.6593 | 10.0057 | 66354.7132 |
| Lowpass25 | 同结构分支，低通 RGB 输入 | 9.8509 | 21.2414 | 8.2226 | 56140.2192 |

证据：[固定版本比较表](https://github.com/xizaoge-shuai/YOLO-Master/blob/2b13720c718dbd14f99d4e722671d930a261997e/research/experiments/detail-probe/job-20261022-hjP4y3Lf/complete/comparison.txt)、[Detail 汇总](https://github.com/xizaoge-shuai/YOLO-Master/blob/2b13720c718dbd14f99d4e722671d930a261997e/research/experiments/detail-probe/job-20261022-hjP4y3Lf/detail25/train/detail25-s0/summary.json)、[Lowpass 汇总](https://github.com/xizaoge-shuai/YOLO-Master/blob/2b13720c718dbd14f99d4e722671d930a261997e/research/experiments/detail-probe/job-20261022-hjP4y3Lf/lowpass25/train/lowpass25-s0/summary.json)。

Detail−Blank 为 +2.3469 AP，Detail−Lowpass 为 +2.4944，Detail−Views 为 +2.2541；均为 seed0 开发验证观测。RGB/低通各增加 4672 参数，低通先 area 缩至40×40，再双线性回640×640；单个低通对照不足以唯一归因于高频或小目标定位。

成本边界：共享 GPU 的时间只作运行记录。Detail 观察训练时间约为 Blank 的2.06倍；q25 只表示教师图像查询量，不等于节省75%训练时间。RGB 臂还读取全部训练 RGB。Detail/Lowpass 各有72786次训练提取调用，不能与161775张查询图像混为一个计数。Views 依赖缓存135.1833 GiB，Blank/Detail/Lowpass 为46.3177 GiB，均含相应训练和验证缓存；嵌套读取计时不得再次加到墙钟。

跨轮比较须核对 checkpoint、实际加载源码、`source_hook` 和评估指纹。本轮Transport25确有完整训练；它与旧表AP相同，不能仅凭同值推断复用了旧权重，也不增加独立训练种子数。旧 Mix25/Online100 属另一轮历史参照，不能把 Detail−旧Online 的数值差写成同次受控收益或将旧时间混入本轮速度比较。

## 5. 完整检测器参照

| 检测器 | 最终参考 AP | AP50 | AP75 | 训练记录 |
| --- | ---: | ---: | ---: | --- |
| YOLO-Master-N | 14.8749 | 27.4598 | 13.8138 | seed0；100 unique epochs，存在续训/重复记录 |
| YOLO11-N | 14.7748 | 27.5637 | 13.5523 | seed0；100 unique epochs |
| YOLO12-N | 14.3876 | 26.9711 | 13.3172 | seed0；100 unique epochs |

用户原始日志见聊天 L29530–29549、L29715–29721、L29992–29999：六次 last/best 评估通过 checkpoint SHA256、548图覆盖和参考版本检查。Master-N 有198行、100个独立epoch，其中98个epoch重复且值不同；不得按198epochs计算，也不能据此给出干净的成本排名。完整检测器与冻结 DINO 系统的结构、初始化、训练配方及训练/推理成本应各列明。

2026-11-07 已只读核验原服务器保留的三个 `args.yaml`，seed 均为 0。Master 的配置另有从 `last.pt` resume 的记录，进一步确认其训练预算来源需单独处理。原路径记录位于 `runs/paper/detector-baselines/job-start-EQA3xAp5/train/{master-n,yolo11-n,yolo12-n}/args.yaml`；没有改写旧配置。

## 6. 特征增强 pilot：10配置×3种子

VisDrone400/100、100epochs、seeds0/1/2、mean_value；以下为内部诊断 AP 的均值±样本SD，均来自实际30个完成训练。各配置及较差强度全部保留。

| 配置 | Best AP | Final AP |
| --- | ---: | ---: |
| cache1 | 2.5694 ± 0.0597 | 1.8258 ± 0.1988 |
| cache2 | 3.6405 ± 0.2135 | 2.8434 ± 0.1764 |
| frofa-c1-d01 | 2.7640 ± 0.1555 | 2.1806 ± 0.1343 |
| frofa-c1-d03 | 2.4675 ± 0.1952 | 1.9620 ± 0.1250 |
| frofa-c2-d01 | 3.5752 ± 0.1010 | 3.0076 ± 0.1304 |
| frofa-c2-d03 | 3.3241 ± 0.1564 | 2.8405 ± 0.1824 |
| transport-c1 | 3.1775 ± 0.0903 | 2.7574 ± 0.0267 |
| loffta-c1-n005 | 3.2853 ± 0.2950 | 2.9942 ± 0.2228 |
| transport-c2 | 3.7666 ± 0.0810 | 3.4723 ± 0.1805 |
| loffta-c2-n005 | 3.7675 ± 0.0161 | 3.4479 ± 0.1225 |

数值据[逐run比较](experiments/feature-baselines-20260925-job-6JnMcLAf/original/comparison.json)汇总；[归档核验](experiments/feature-baselines-20260925-job-6JnMcLAf/README.md)检查各run的100行曲线、配置和源码，未独立从预测重算AP。验证集也用于选择best和增强强度，因此属于探索。所有run零在线骨干调用；复用缓存的额外构建为0，cold-start估算另收费一次历史训练缓存构建，不含既有验证缓存构建，不能与后续全量成本直接拼接。

## 7. 预算校正及查询策略

预算pilot为400/100、100epochs、seeds0/1/2，共6配置×3个完成训练；同预算共享查询与增强。下表为内部诊断AP。Rec即历史Correct。

| 配置 | Best均值±SD | Final均值±SD |
| --- | ---: | ---: |
| Mix10 | 3.6920 ± 0.2844 | 3.2369 ± 0.2337 |
| Transport10 | 4.0984 ± 0.1540 | 3.7995 ± 0.1007 |
| Rec10 | 3.8714 ± 0.1812 | 3.6928 ± 0.2316 |
| Mix25 | 4.1635 ± 0.2744 | 3.8934 ± 0.3361 |
| Transport25 | 4.5306 ± 0.1011 | 4.2418 ± 0.2094 |
| Rec25 | 4.4190 ± 0.2736 | 4.2809 ± 0.2778 |

证据：[18-run结果与成本表](experiments/budget-corrector-job-start-7U9Q7DSc/summary-table.md)、[原始比较](experiments/budget-corrector-job-start-7U9Q7DSc/original/comparison.json)。Transport25有异常慢run，均值492.0s、中位数421.4s，不能只选中位数宣称速度。

| 后续探索 | 已完成证据 | Final诊断AP均值 / 边界 |
| --- | --- | --- |
| IDGate25 | 三种子用户原始日志 L8751–8761 | 4.0630 |
| IDFG25 | 同上；恒等门控＋前景损失 | 4.2384 |
| Severity25 | 三种子日志 L10049–10067 | 3.6190；Random25为3.8934 |
| Loss-guided25 | 三种子日志 L15064–15077 | 3.8673；额外routing均值136.7s |
| Random Mix50 pilot | 三种子日志 L15513–15519 | 4.2325 |
| 全量 gate25/fg25/gatefg25 | 有[固定消融协议](IDENTITY_FOREGROUND_ABLATION_PLAN.md) | 不能仅凭计划、脚本或目录认定完成 |
| Anchor25/Task25 | 助手曾报10.1128/10.0809，聊天 L30474–30475 | 缺该处用户可读原始日志；待实际产物核验后入正式结果表 |

本节查询策略/ID候选的原始文字可查[聊天导出](../ChatGPT_答辩PPT与稿子撰写_20261106-2309.md)；完整轻量归档、实际args、预测和源码指纹仍需拉齐。不要把不同前景mask实现的pilot与全量FG视为同配置。

## 8. 其他数据及机制证据

VOC子集 q0/q25/q100 的最终内部AP分别为22.7129/24.1568/27.1814；seed0、100epochs、batch8、640。观察wall分别10800.7/11999.4/6763.2s，未证明q25比全在线更快。原始日志L25525–25536。

COCO val1000：cached latent最佳0–1 mAP50–95为0.006733611853143846；YOLO26n-scratch最佳内部mAP50–95日志为0.00863。二者均100epochs；前者seed0见L5795–5808，后者完成日志L6092–6100。不得扩写成完整COCO或YOLO-Master-N基线。

| 机制阶段 | 图像与操作 | 能支持什么 |
| --- | --- | --- |
| [留出特征/任务诊断](experiments/heldout-corrector-job-7c6i3o4w/analysis-report.md) | 548验证图，5480 teacher images，零更新 | 非恒等视角特征/固定头损失改善；不是AP增益 |
| Query-value oracle | 2192变换实例，零训练更新，聊天L12852–12894 | 即时损失选择空间；oracle25不是训练AP oracle |
| [分配干预及头复核](analysis/assignment-replication-20261018/analysis-report.md) | 完整548验证图；改变已有冻结头 | 不是跨骨干、跨数据集或独立训练种子 |
| [空间干预](analysis/spatial-intervention-20261018/analysis-report.md) | 固定128训练图、两冻结头、768+1教师图、零更新 | 填充/内容敏感性；自然区域尚未等token/等能量 |
| [联合原生梯度](analysis/joint-gradient-20261019/analysis-report.md) | 固定128训练图、768+1教师图、零更新 | 三部署候选12分组方差比均>1，未过原门槛；无新AP |
| CacheShift V0 | 548验证图×6视角，3288教师图，聊天L20545–20576 | 自定义变换诊断；旧flip相对MSE0.577536，不能写成精确C2锚点 |

## 9. 仍需补的证据

- 同轮配对种子1/2：Detail、Blank、Lowpass、Views与Transport；核对旧Mix/Online的加载源码与评估后决定是否重跑，不能直接补进误差条。
- 全量FroFA/LOFF-TA检测适配对照，以及C2/Mix25/Online100的当前版本可比性；原论文分类配方复现与检测适配应明确标注。
- 受控等时间/端到端成本：缓存构建、训练、验证、全部RGB读取、教师调用、存储、峰值显存；共享GPU时间不支持速度优越性。
- 第二骨干、正式第二数据集协议、完整COCO；当前VOC子集和两个冻结头诊断不能替代。
- 机制的等token/等残差能量控制及独立评估；已有548开发验证图参与方法选择，不能再称未见测试集。
- 将聊天中的原始run配置、预测/评估、模型哈希和成本补齐到轻量归档；保留Master训练历史冲突、失败与重复来源。台账中的缺失项不写成已完成。

## 10. 四篇论文补验的优先安排

用户在2026-11-07进一步要求：直接复刻论文思想适配检测，不下载作者源码，不新增GPU冒烟或小子集训练。已逐篇读取所附PDF。完整方案见[四篇论文全量协议](PAPER_BASELINE_PROTOCOL_20261107.md)。

新增的28项优先队列保留原20项并去重：先四篇适配seed0，再其余种子、同协议预算参照、零查询与INT4对照，最后接续原五臂种子。已开始的Blank25-s1保留原检查点及源码，从原job继续；不重头训练。具体运行状态以新job为准，本文不会将排队配置计为完成结果。

按用户截图组织完整检测器、缓存刷新、相关方法、空间消融、成本资源五类表。YOLO13没有实现或完成证据，保留缺失。ViT-Adapter文中已经包含SPM-only及空间细节分析，现有浅RGB分支不是凭题目较新就能认定独特的机制；需用本次较强空间适配器比较验证。此处只组织结果与实验，不修改论文。


<!-- results-gap-20261108 -->
## 11. 2026-11-08：三个全量相关方法已完成

观察时间：2026-11-08 19:41:48（UTC+8）。28项队列完成3项；`vit-adapter-cache25-s0`在第54/100轮，未发现失败记录。训练进程在运行，冻结源码无Git修改。

| 方法 | seed | q% | 最终AP | AP50 | AP75 | Detail25−对照AP点 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cache-reuse-c1-int8 | 0 | 0 | 8.0720 | 18.0685 | 6.3517 | +4.2733 |
| frofa-c2-d01 | 0 | 0 | 10.5272 | 22.9663 | 8.3076 | +1.8181 |
| loffta-c2-n005 | 0 | 0 | 9.1234 | 19.9421 | 7.4518 | +3.2219 |

均为6471/548、100轮、batch8、固定最终参考评分；三项各548份预测和最终检查点哈希核对通过，归档均已push。其共享头初始化和参考实现与历史五臂相同。上述与Detail25的差值为跨轮、不同查询预算的描述性参照；Detail25仍只有seed0。原始配置与证据见[本轮核验快照](checks/paper-baselines-20261107/results-gap-20261108.json)及其中每项的归档路径。

当前继续已有28项全量队列，不追加冒烟或小规模训练。优先看ViT-Adapter-style q25最终AP，并补齐候选及关键对照种子；不能用零查询弱对照的差值替代同预算检验。成本仍是共享GPU观察记录，Detail历史训练时间约为Blank的2.06倍。没有改论文或训练方法。


<!-- storage-staging-20261108 -->
## 12. 2026-11-08：分批数据安排与可恢复副本清理

现有完整主实验仍为VisDrone6471/548；服务器COCO train2017为空、val2017为5000图。新增正式COCO2017、VOC07+12、SKU-110K已选定，尚未下载或训练；当前队列继续，先补指标再调优方法。见[数据与证据计划](DATASET_EXPANSION_PLAN_20261108.md)。

清理仅涉及已与现有文件/保留标注ZIP逐字节对齐的COCO临时解压副本，共122225文件；观察释放530046976字节。未删除数据集、权重、特征银行或唯一标签。[清理证据](checks/paper-baselines-20261107/cleanup-result-20261108.json)含完整清单的SHA256及运行目录位置。没有新增AP结果、GPU测试或方法改动。

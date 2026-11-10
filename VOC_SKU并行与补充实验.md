# VOC、SKU并行流水线和补充实验

2026-11-10。当前执行平台yang109。保持100轮、现有增强/优化器和固定最终轮端点，先跑实验数据，再种子与调优。

## 并行范围

VOC07+12已完成全量准备，SKU-110K fixed继续后台下载。每个数据集有独立状态、断点文件、原始标注、下载来源和完成封印。只有完整划分与全部文件通过检查、生成complete.json后，训练流水线才继续。

GPU1驱动报告GPU requires reset，目前不选用。两个新数据集流水线与新增诊断共用健康GPU0的准入队列，最多同时运行两个新增计算阶段：首个要求至少8GiB空闲显存，第二个要求至少16GiB空闲显存，否则后台等待；现有VisDrone训练不重启。下载、CPU评价、状态监控可以并行，GPU并行发生在同一张卡上。GPU0还有其他用户工作，耗时是观察记录，不是受控速度benchmark。

当前下载经Windows Clash7897和SSH反向127.0.0.1:7907加速，仅数据准备进程使用代理。请保持Clash和这条SSH隧道运行；关闭电脑/隧道会影响下载。训练使用本地权重和数据，不依赖该代理。

## 固定矩阵

每个新数据集先运行C1、C2、Mix25、Online100、FroFA、LOFF-TA、CacheReuse-INT8、Adapter25，再Transport25、Blank25、Detail25、Lowpass25。共12个seed0配置；相关论文均为明确标注的思想适配，不是原论文源码复现。额外视角、INT4、零查询Adapter、seed1/2先延后。完整YOLO-Master/YOLO11/YOLO12在新数据集上的训练未纳入这12项，不将本批完成等同全部论文实验完成。

VOC使用16551训练/4952测试、20类，主要指标是VOC2007十一点AP50，difficult匹配忽略。SKU预期8219训练/588验证/2936测试、1类，主要指标为标准COCO maxDet100的bboxAP，另报密集maxDet500；实际计数以固定发布包检查结果为准。两者都只评固定最终100轮测试端点，不在测试集上挑最好轮或调超参数。

新文件crossdataset_suite.py复用已有模型/增强/损失，新增可变类别与正式数据输入、mmap银行、独立评估和断点证据。缓存恢复时发现缺失前缀/rename-seal窗口会保留文件并明确报错，避免悄悄补零覆盖。物理教师调用与逻辑预算分开记录，量化构建不重复计入冷成本。

首次VOC快照因误将官方训练集内部3对重复JPEG当成跨划分泄漏而停止，未构建缓存或训练。已从原ZIP核验：sample ID不重复、训练/测试内容不重叠，保留官方16551训练样本与各自标注；新校验只拒绝跨划分重复内容及重复ID，训练内部重复在provenance显式记录。失败job保留，新job重新快照；sealed数据没有去重或改写。

## 看下载、训练和结果

```bash
cd ~/YOLO-Master-paper-baselines-20261107
DATA_ROOT="$(cd ../datasets-extension-20261110 && pwd)"
for DATA in voc0712 sku110k; do
    echo "$DATA"
    cat "$DATA_ROOT/$DATA/status.txt"
    tail -n 5 "$DATA_ROOT/$DATA/prepare.log"
done
```

两个流水线启动后：

```bash
for DATA in voc0712 sku110k; do
    JOB="$(cat "runs/paper/crossdataset/latest-$DATA-job.txt")"
    python online_experiments/crossdataset_campaign.py status --job "$JOB"
    tail -n 8 "$JOB/pipeline.log"
done
```

WAITING_DATA是等待完整数据封印；WAITING_GPU是等待健康GPU准入；BUILDING_CACHE、TRAINING、VALIDATING分别对应真实阶段；SUCCEEDED要求12个最终端点通过校验。PID退出不能当作成功。已存在训练子进程或任务锁时新入口拒绝重复启动。

## 已补的VisDrone离线分析

对8个方法各548张最终预测完成尺寸/逐类别/错误/曲线分析。原官方AP逐项重算一致；额外尺寸和TIDE另用声明的辅助协议，不混为VisDrone官方AP。

| 方法 | 辅助COCO-style AP_S@500 |
| --- | ---: |
| Transport25 | 3.0694 |
| Blank25 | 3.1329 |
| Views25 | 3.4000 |
| Detail25 | 5.4259 |
| Lowpass25 | 3.1494 |
| Mix25 | 4.2209 |
| Online100 | 4.2094 |
| Adapter25 | 9.0101 |

Detail相对Blank的小目标AP提高2.2930点，但仍低于Adapter。不能仅凭这些结果声称所有基线领先。这里均为seed0，不生成独立训练重复的误差条。

Detail相对Blank的中目标AP提高0.9358点，大目标AP下降0.6321点。TIDE的Miss dAP从6.8354降到5.2448，Loc dAP仅从7.8006降到7.7085；分类和背景错误的dAP升高。支持讨论小目标与漏检收益，同时保留大目标和误检代价。

首次离线分析发现历史Detail导出中有37个负宽/高框，已保留失败记录。修订辅助协议将负宽/高置零并保留其score/类别/检测数，使其仍作为退化预测参与评价；原始预测文件和原官方评分输入不改。四图及CSV来自全548分析，不重训模型。

```bash
ANA="$(cat runs/paper/evidence-analysis/latest-job.txt)"
cat "$ANA/status.txt"
cat "$ANA/summary.csv"
cat "$ANA/tide-errors-auxiliary.csv"
```

新P3频谱、真实新图推理及几何特征/损失诊断共三个GPU阶段，完成状态以其job为准：

```bash
GJOB="$(cat runs/paper/evidence-gpu/latest-job.txt)"
cat "$GJOB/status.txt"
tail -n 8 "$GJOB/pipeline.log"
```

features覆盖4个固定最终模型各548图，使用真实验证银行；latency则为每个方法各548张新图输入，独立执行DINO前向，另计每方法20张预热，共2192实测＋80预热教师图像，不用缓存替代推理。当前检测头采用端到端解码/top-K，独立NMS不适用；推理后处理测置信度筛选和原图坐标恢复，不虚构NMS耗时。

geometry使用548图×6视角，共3288次新教师图像/调用，四个固定模型共享这批诊断特征；identity/hflip是真C2锚点零误差对照，其余四视角为0.85/0.95/1.05/1.15缩放。统计原DINO误差、各模型自身approx/exact P3误差、检测损失差及GT裁剪数，不更新权重，不产出变换后的官方AP。Detail加性RGB在同模型P3绝对差中抵消，不能据此声称恢复了DINO表示；旧校正器CacheShift也不充当当前候选的结果。

## 文件与论文结构

- online_experiments/prepare_detection_datasets.py：数据准备，不训练模型。
- online_experiments/crossdataset_suite.py：新正式数据训练、缓存、评价。
- online_experiments/crossdataset_campaign.py：12项矩阵、健康GPU准入、归档commit/push。
- online_experiments/analyze_detection_evidence.py：原最终预测的CPU分析。
- online_experiments/spatial_feature_analysis.py：固定模型的特征/真实推理测量，单独记录教师计算。
- online_experiments/geometric_evidence.py：六视角全548图的特征/固定检测头损失诊断。
- online_experiments/run_spatial_evidence.py：上述三阶段的GPU准入、日志、完整覆盖校验与轻量归档。
- research/PAPER_EVIDENCE_PLAN_20261110.md：章节、五类表、四张主图和各指标/数据依赖。

沿用第3节缓存增强问题、第4节实际方法、第5.2节baseline、第5.3节成本、第5.4节消融机制、第5.5节数据/骨干适用范围、第5.6节困难场景与失败的结构。优先补本表中的证据，不增加与主张无关的分类、分割或功耗实验。全COCO、第二骨干、独立种子和受控等时间仍未被本批替代。

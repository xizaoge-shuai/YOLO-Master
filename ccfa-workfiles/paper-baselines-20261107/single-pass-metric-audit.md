# 保存预测的一次指标收集审查

核查日期：2026-11-14。范围为本地同步代码与已有轻量证据；未重新训练、推理或运行检测评估。本记录不改变正在运行的训练或指标程序。

## 已有自动覆盖

VOC/SKU的七方法队列均由`online_experiments/metric_bundle_campaign.py`等待固定seed0、final100完整测试端点，再从原预测生成CPU指标包。`complete_metric_bundle.py:58`的一次COCOeval evaluate/accumulate同时覆盖maxDet1/10/100/500。

| 项目 | 现有覆盖与边界 |
| --- | --- |
| AP、AP50、AP75、AP small/medium/large | 已自动导出；按数据集原生或辅助协议分别列明 |
| AR与maxDet | AR@1/10/100/500及各尺度AR已导出；SKU100/500分列 |
| 类别、尺度、GT数量 | `per-class.csv`、`per-area.csv`、`object-counts.csv` |
| PR | 101个召回格点的插值precision和score；CSV提供IoU0.5/0.75、maxDet100/500，NPZ保留完整张量。它不是confidence-sweep P/R/F1曲线 |
| 训练曲线 | 真实loss、时间、学习率等；未逐轮验证的占位0不当作AP |
| 资源 | 保存的训练/过程/评估时间、缓存构建与量化、存储、查询图像/调用及物理journal统计、参数和峰值allocated VRAM；原记录缺失项保留null |
| TIDE | VisDrone七方法已有独立CPU补指标并完成。VOC/SKU的现有bundle没有TIDE |

VOC原生主指标为inclusive坐标、IoU严格大于0.5的十一点mAP50；VOC的COCO-style尺度/AP75等只是辅助结果。现有COCO转换采用半开坐标并将difficult映射为iscrowd，不能把辅助AP50冒充原生值。SKU保留原封存manifest及作者loader边界处理；读取实际ignore字段，不新增忽略区域。

保存预测的置信度下限0.001及全图top500上限已经固定。低于下限或超出上限的检测无法事后恢复。maxDet扫描不需要再次推理。

## TIDE结论

本轮优先完成新增训练并发和现有完整指标包，不立即增加VOC/SKU TIDE进程。SKU密集预测的TIDE对象及错误oracle可能占用较多主机内存；本地未取得它的实测峰值和运行时间，不能将其描述为可忽略开销。VisDrone现有TIDE已能支持检测错误分析。

如后续仍需要VOC/SKU TIDE，可靠的最小实现是独立CPU sidecar及独立封存目录，复用已完成端点和保存预测，不修改现有bundle或其哈希契约。建议只算TIDE@500一次；AP/AR@100/500已由原bundle提供。

- 复用`metric_bundle_campaign.py:131`的输出解析、ready/verify_bundle和同一归档锁；每个端点只处理一次，已封存结果先验哈希后跳过。
- 复用`crossdataset_suite.coco_payload`的GT/预测转换语义。现有`analyze_detection_evidence.py:221`的`tide_data`硬编码VisDrone十类，不能直接用于VOC二十类；需一个通用Data适配器。
- 保留现有`evaluate_tide`配置：foreground IoU0.5、background IoU0.1、BOX模式、Data max_dets500、同类ignore区域；保留非ignored错误数与库原始错误数，oracle dAP50不求和。TIDE AP属于独立辅助协议，不要求与COCO或VOC原生AP数值相等。
- VOC difficult进入同类ignore；SKU读取封存manifest实际iscrowd/difficult。SKU单类别的分类错误项目解释力有限。
- 单进程、CUDA不可见、BLAS/OpenMP单线程、nice10，主机MemAvailable至少24GiB时才启动一项；避免与现有大JSON CPU评估同时启动。该阈值是资源准入条件，不是经测量的TIDE峰值保证。
- 记录输入/输出和依赖版本哈希。未封存的失败结果保留且不自动反复重跑；不改原生主指标。

## 不需重训练或推理的后续汇总

新端点指标包封存后，`export_map_summary.py`与`export_cost_accuracy.py`只读现有数据即可刷新mAP及成本表。资源summary中已记录、但当前精简导出未列出的主机RSS或dequant时间，也可通过独立汇总补列，不需要重新测量。

现有记录不能重建历史能耗、统一训练FLOPs、独占GPU成本、真正等时间AP或未保存的逐轮AP。P95需要原逐图计时样本；当前只保存均值的直接检测器不能从均值还原P95。缓存头的测试时间不能替代完整图像到检测的推理延迟。上述缺项继续留空，不为补表重复已完成实验。

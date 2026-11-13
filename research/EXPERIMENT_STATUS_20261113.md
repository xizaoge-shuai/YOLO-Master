# 实验进展：2026-11-13 晚间

执行服务器yang109，项目`/mnt/data/users/zhjia/YOLO-Master-paper-baselines-20261107`。本次核查结果、恢复数据下载、补充CPU并行准备与指标收集，没有重新训练、追加模型推理或修改冻结训练源码。

## 当前训练与指标

后续mAP专项核查：VOC Master推进到第97/100轮，SKU FroFA已完成45/100轮；COCO原始数据准备完成、训练仍未启动。当前所有可用mAP及未完成项见`research/metrics/map-summary/map-results.csv`和`coverage.csv`，旧VOC C1/C2/Mix25与SKU C1作为历史对照单独保留。

本轮每数据集七方法：真实YOLO-Master-N、真实YOLO12-N、FroFA/LOFF-TA/Cache and Reuse INT8/ViT-Adapter四种思想适配、候选Detail25。固定seed0、100轮，未追加独立种子或扩展消融。

| 数据集 | 本轮七方法完成情况 | 正在执行 | 说明 |
| --- | --- | --- | --- |
| VisDrone | 7/7，复用已完成结果 | 无重复训练 | 完整6471训练/548验证，历史控制与分析继续保留 |
| VOC0712 | 0/7 | YOLO-Master，第97/100轮 | 完整16551训练/4952测试；历史C1/C2/Mix等结果仍保留，0/7仅指本轮七方法 |
| SKU110K | 2/7 | FroFA，已完成45/100轮 | 真实YOLO12和YOLO-Master已完成；完整8219训练/2936测试 |
| COCO2017 | 未启动训练 | 完整数据已准备完成 | 118287训练图/5000验证图，PREPARED_COCO；训练继续安排在最后 |

CPU指标后台仍运行，14项中2项完成、0项失败，其余等待原训练端点。两个已完成SKU指标包已包含原生主指标校验、分类/尺度结果、AR、PR张量、实际训练曲线与资源记录，没有第二次模型预测。

## 新增SKU结果

固定最终100轮，完整2936测试图，0–100尺度。AP@100和密集AP@500是不同maxDet设置，分别报告。

| 方法 | AP@100 | AP50@100 | AP75@100 | dense AP@500 | AP50@500 | 实际总参数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO-Master-N | 40.3206 | 62.5596 | 47.0385 | 53.8890 | 88.5268 | 2812651 |
| YOLO12-N | 40.3693 | 62.5773 | 47.1342 | 54.1085 | 88.8811 | 2568243 |

两项native_consistency为PASS，输入输出封印核对通过。该表是每方法一次运行，不作显著性结论。SKU候选Detail25尚未完成，当前不能报告其与这两个检测器的性能差距。旧C1的AP@100=34.2675、AP@500=45.6275仅作为已有缓存参照，不能替代最终候选。

初次核查时曾估计VOC当前Master剩余约2小时、SKU当前FroFA约7小时；这是先前进度的历史估计。最新进度以上表为准，共享GPU速度会变化，不能据旧估计判断整条队列完成时间。VOC完成Master后会继续YOLO12；SKU完成FroFA后继续LOFF-TA。

## 与其他论文思想baseline的已有比较

VisDrone七方法的CPU完整补指标现已全部完成，7/7成功、0失败。另新增[mAP口径和完整表](metrics/map-summary/README.md)：`map-results.csv`从已封存结果汇总mAP50、mAP50–95，并区分VisDrone参考类别等权、原参考加权与COCO-style辅助协议。SKU现有COCO AP本身已是mAP；VOC原生十一点指标为mAP50。汇总脚本不调用模型或评估器，原结果不覆盖。

VisDrone完整548张开发集的保存结果已完成，0–100尺度。以下是相关论文思想的检测适配，不是其原论文源码复现。候选方法与强Adapter的负差距保留，不按结果筛baseline。

| 方法 | 已测AP | Detail25减该方法AP |
| --- | ---: | ---: |
| FroFA适配 | 10.5272 | +1.8181 |
| LOFF-TA适配 | 9.1234 | +3.2219 |
| Cache and Reuse INT8适配 | 8.0720 | +4.2733 |
| ViT-Adapter适配 | 16.5855 | -4.2402 |
| Detail25 | 12.3453 | 0 |

真实Master和YOLO12的已有VisDrone AP分别14.8749、14.3876，不能用它们的SKU结果替代VisDrone比较。历史检测器训练轨迹与新VOC/SKU的严格100唯一epoch合同分别审计，保存分数不代表所有训练成本已可公平比较。完整七项来源表位于`research/checks/minimal-priority-20261112/reused-visdrone-core-results.csv`。

新增CPU并行准备两类工作：预构建VOC/SKU所需的物理INT8缓存；从现有VisDrone预测补齐缺少的分类、尺度、PR/张量和TIDE证据。已完成的八臂分析保留，Adapter/Detail已有TIDE直接复用。两项GPU训练继续，VOC/SKU每项最终预测后的CPU指标后台继续运行。

两项已实际后台启动：INT8准备PID1276050，首次核查为`PREPARING_CPU`、VOC已处理876/16551图；VisDrone补指标PID1276051，首次核查为`RUNNING_CPU`、当前真实Master。两项均独立后台执行，完成后自动轻量归档、commit/push，不需要Codex持续在线监控。源码已提交并push：`1f8fa1f`。输出分别位于`runs/paper/priority-int8/job-20261113-cpu/`和`runs/paper/visdrone-core-metrics/job-20261113-single-pass/`，启动实测快照见`research/checks/status-20261113/parallel-cpu-launch.json`。

资源核查：GPU0利用率100%，另有其他用户进程占24660MiB，剩余约14.4GiB；GPU1异常。CPU约85%空闲，可用内存约47GiB、I/O等待0–1%。本批新增工作使用CPU低优先级，不绕过现有两路GPU准入增加第三项训练。INT8预构建预计额外存储约45.9GB，原FP16银行保留，未来训练自动复用；原构建耗时仍计入冷成本。

## 下载与Git

发现原反向代理7908已停止监听，COCO保留的13,715,363,183字节partial未删除。已恢复电脑Clash7897到服务器7908的SSH反向隧道（本地进程2284），服务器数据准备进程1264009继续断点下载。核查时partial已增长到13,801,694,575字节。COCO没有开始训练。

SKU两个新检测器及指标已自动commit；其中Master原结果与指标的push曾超时，服务器文件和本地提交均完整。本次统一推送会包含这些父提交，不重复训练或重新计算。原超时回执保留作为历史记录。

## 看进度

```bash
cd ~/YOLO-Master-paper-baselines-20261107
PY=/mnt/data/users/zhjia/miniconda3/envs/yolomaster/bin/python
PRIORITY=runs/paper/priority/job-20261112-single-seed

for DATA in voc0712 sku110k; do
    "$PY" online_experiments/priority_single_seed_campaign.py status --job "$PRIORITY/$DATA"
done

"$PY" online_experiments/metric_bundle_campaign.py status \
  --job runs/paper/metric-bundles/job-20261112-single-pass

# CPU预构建已有INT8 baseline缓存
"$PY" online_experiments/prepare_priority_int8.py status

# VisDrone七方法补全指标
cat runs/paper/visdrone-core-metrics/job-20261113-single-pass/status.json
```

训练和CPU汇总独立后台执行；退出Codex不会停止训练。COCO下载依赖电脑、VPN、Clash及SSH隧道在线。详细快照见`research/checks/status-20261113/snapshot.json`。

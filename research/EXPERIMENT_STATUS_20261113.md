# 实验进展：2026-11-13 晚间

执行服务器yang109，项目`/mnt/data/users/zhjia/YOLO-Master-paper-baselines-20261107`。本次只核查结果、恢复数据下载和同步Git，没有重新训练、追加推理或修改冻结训练源码。

## 当前训练与指标

本轮每数据集七方法：真实YOLO-Master-N、真实YOLO12-N、FroFA/LOFF-TA/Cache and Reuse INT8/ViT-Adapter四种思想适配、候选Detail25。固定seed0、100轮，未追加独立种子或扩展消融。

| 数据集 | 本轮七方法完成情况 | 正在执行 | 说明 |
| --- | --- | --- | --- |
| VisDrone | 7/7，复用已完成结果 | 无重复训练 | 完整6471训练/548验证，历史控制与分析继续保留 |
| VOC0712 | 0/7 | YOLO-Master，第91/100轮 | 完整16551训练/4952测试；历史C1/C2/Mix等结果仍保留，0/7仅指本轮七方法 |
| SKU110K | 2/7 | FroFA，已完成33/100轮 | 真实YOLO12和YOLO-Master已完成；完整8219训练/2936测试 |
| COCO2017 | 未启动训练 | 完整数据断点续传 | 下载完成及全图核验后才算数据准备好，继续安排在最后 |

CPU指标后台仍运行，14项中2项完成、0项失败，其余等待原训练端点。两个已完成SKU指标包已包含原生主指标校验、分类/尺度结果、AR、PR张量、实际训练曲线与资源记录，没有第二次模型预测。

## 新增SKU结果

固定最终100轮，完整2936测试图，0–100尺度。AP@100和密集AP@500是不同maxDet设置，分别报告。

| 方法 | AP@100 | AP50@100 | AP75@100 | dense AP@500 | AP50@500 | 实际总参数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO-Master-N | 40.3206 | 62.5596 | 47.0385 | 53.8890 | 88.5268 | 2812651 |
| YOLO12-N | 40.3693 | 62.5773 | 47.1342 | 54.1085 | 88.8811 | 2568243 |

两项native_consistency为PASS，输入输出封印核对通过。该表是每方法一次运行，不作显著性结论。SKU候选Detail25尚未完成，当前不能报告其与这两个检测器的性能差距。旧C1的AP@100=34.2675、AP@500=45.6275仅作为已有缓存参照，不能替代最终候选。

按最近训练速度，VOC当前Master剩余约2小时、SKU当前FroFA剩余约7小时，随后还要最终预测与CPU评分。该估计只针对当前方法，不是整条队列完成时间；共享GPU速度会变化。VOC完成Master后会继续YOLO12；SKU完成FroFA后继续LOFF-TA。

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
```

训练和CPU汇总独立后台执行；退出Codex不会停止训练。COCO下载依赖电脑、VPN、Clash及SSH隧道在线。详细快照见`research/checks/status-20261113/snapshot.json`。

# 校园两路 + AutoDL两路：2026-11-14接手

2026-11-15更新：完整迁移包已封存；两个源暂停进程已退役，资源已释放。后台传输、AutoDL自动等待/续训及回传客户端均已实际启动，当前为WAITING_TRANSFER；校园VOC FroFA和SKU INT8已完成，SKU继续Adapter，VOC接续其余方法。最新实时命令与解释见[今日进展](实验进展_20261115.md)，以下原暂停现场描述保留为历史。

用户最新要求是停止校园额外两路，并在AutoDL续跑它们。校园保留FroFA/INT8；迁移VOC LOFF-TA、SKU Detail25，固定seed0、100轮、batch8、640和完整原划分。

当前已确认：两迁移进程处于SIGSTOP，完整检查点为LOFF9轮/Detail17轮；原FroFA/INT8继续。AutoDL连接成功且缺失依赖已安装，GPU/环境导入验证、数据复制与新训练分别记录。不能把等待/复制状态当成训练已经开始。

## 校园服务器

项目`/mnt/data/users/zhjia/YOLO-Master-paper-baselines-20261107`。

```bash
cd ~/YOLO-Master-paper-baselines-20261107
J=runs/paper/migration-autodl/job-20261114-two-lanes
cat "$J/paused.json"
test ! -f "$J/transfer-state.json" || cat "$J/transfer-state.json"
test ! -f "$J/return-status.json" || cat "$J/return-status.json"
tail -n 20 "$J/prepare-bundle.log"
```

校园原训练日志仍在`runs/paper/priority/job-20261112-single-seed/{voc0712,sku110k}/train-<method>.log`。新接管器将明确跳过迁移项，避免再次启动。源暂停现场退役前须完成备份/清单封存，退役记录为`$J/retirement.json`。不要手动恢复旧四路监督器。

## AutoDL

SSH入口仍为`ssh -p 37540 root@connect.weste.seetacloud.com`。密码不写入文档、脚本或argv。

新工作盘根目录`/root/autodl-tmp/campus-mirror`；旧绝对路径`/mnt/data/users/zhjia`只通过新建符号链接映射到此目录，不覆盖已有数据。新运行目录如下：

```bash
M=/root/autodl-tmp/campus-mirror
J="$M/.migration/job-20261114-resume"
cat "$J/sku110k/status.json"
cat "$J/voc0712/status.json"
tail -n 20 "$J/sku110k/resume.log"
tail -n 20 "$J/voc0712/resume.log"
nvidia-smi
```

状态顺序是WAITING_TRANSFER→UNPACKING→VERIFYING→TRAINING→SUCCEEDED，失败保留FAILED及日志，不自动重做训练。SKU数据先到即可独立续训，无需等VOC传完。目标容器实际62GiB内存/16CPU、GPU32GiB、数据盘450GiB；宿主`free`显示的512GiB不是容器额度。

## 一次预测收齐指标

每项原训练器只保存final100完整测试预测。目标成功后回传器逐SHA获取结果，保留校园原暂停输出，再发布完整canonical端点；原CPU队列从这些预测计算mAP/AP50/AP75、尺度/类别、AR/PR及已记录资源，不重复推理。

```bash
cd ~/YOLO-Master-paper-baselines-20261107
PY=/mnt/data/users/zhjia/miniconda3/envs/yolomaster/bin/python
"$PY" online_experiments/metric_bundle_campaign.py status \
  --job runs/paper/metric-bundles/job-20261112-single-pass
```

迁移保留原checkpoint、优化器/调度器、RNG、schedules和全部teacher查询日志。未完整保存的半轮需从上个完整轮重放，实际查询成本不能抹去。原环境移存environment-origin.json，新environment.json记录真实AutoDL环境。跨硬件不保证逐位一致，累计墙钟时间是混合硬件观测值，不能包装成受控提速。

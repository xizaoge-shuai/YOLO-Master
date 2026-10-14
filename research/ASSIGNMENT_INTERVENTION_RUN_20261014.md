# 2026-10-14：特征与标签分配交叉干预的实际运行

本轮已经在服务器启动，不是待执行建议。后台任务：

`runs/paper/assignment-intervention/job-20261014-EgMkSelF`

实验代码及预先固定的方案提交为 `ad972524756419a86f9bfeb973e9edcc5560f5ad`。
复用 full VisDrone 训练结果：6471 张训练图、548 张验证图；固定 Online100
seed-0 最后一个 epoch 的检测头，对 Transport 和既有 Rec 校正器做机制诊断。
本轮没有更新权重，尚不能证明新方法的 AP 收益。

## 已通过的真实检查

2 张训练图、每图 6 个视角、12 次真实骨干特征提取、384 个干预单元。
原生损失与诊断实现的最大绝对差值为 0；缓存身份检查最大相对 L2 为
0.000034996；检测头和校正器状态均未改变。该结果只验证执行正确性，
两张图不构成机制结论。初始化失败日志已保留并提交。

## 后台顺序

1. 2 张图冒烟检查：已完成。
2. 固定随机种子 20261014 选择的 128 张训练图：正在运行。
3. 全部 548 张既有验证图：上一阶段通过后自动开始。

每张图均包含不翻转/翻转，以及 1.0、0.85、1.15 三个尺度。
分别交换真实/近似特征与真实/近似分配，保留完整四格。
原生软目标和固定公共分母的硬目标分开报告；分类输出梯度、框回归输出梯度、
前景匹配及目标覆盖分开统计。这里的目标尺寸分组不是官方 AP_S。

GPU1 实际 CUDA 分配失败；当前使用 GPU0、小批量 1，与其他计算共享 GPU。
初期约 20 秒/图，全部诊断约需 3.5–4 小时，不能拿该耗时作效率比较。
没有新建大特征缓存，没有删除原实验或干扰其他进程。

## 查看命令

在服务器执行：

```bash
cd ~/YOLO-Master
bash online_experiments/run_assignment_intervention.sh status
```

实时日志：

```bash
JOB="$(cat ~/YOLO-Master/runs/paper/assignment-intervention/latest-job.txt)"
tail -f "$JOB/job.log"
```

`job.log` 记录阶段起止；细进度在 `development.log` / `confirmation.log`。
查看最终结果与归档提交：

```bash
JOB="$(cat ~/YOLO-Master/runs/paper/assignment-intervention/latest-job.txt)"
cat "$JOB/status.txt"
if [[ -f "$JOB/confirmation/comparison.txt" ]]; then
    cat "$JOB/confirmation/comparison.txt"
elif [[ -f "$JOB/development/comparison.txt" ]]; then
    cat "$JOB/development/comparison.txt"
else
    tail -n 10 "$JOB/development.log"
fi
if [[ -f "$JOB/archive-status.json" ]]; then
    cat "$JOB/archive-status.json"
fi
```

独立归档进程验证完整样本计数、冻结状态、五份原始输出文件的 SHA256，
再按阶段保存小型结果表并 commit。原始逐图 CSV 保存在运行目录。
若发现其他人正在编辑进展文件或暂存其他内容，归档会停止并记录原因，
不会将其混入提交；这不会停止计算任务。没有 push。

## 读结果时的边界

四格代码中首字母是预测特征来源，第二个是监督分配来源：E=Exact，A=Approximate。
核心比较是 AA 与 AE，同时保留 EE、EA 两个对照。
即使替换为真实特征的分配后梯度更接近 EE，也只是冻结检测头下的局部干预证据，
仍需同在线查询预算的训练实验验证 AP。548 张验证图此前已参与研究分析，
不称作未触碰测试集。本轮只测一个冻结头和一套骨干，不宣称跨模型泛化。

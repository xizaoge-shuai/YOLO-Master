# D1 特征增强基线：2026-09-25

本轮 10 个配置、3 个种子均完成，每组 100 epochs。数据仍为 VisDrone 400/100 小样本划分，DINOv3 ViT-S/16、640 输入、三层特征和 mean_value 检测头。FroFA / LOFF-TA 为核心思想的检测适配，不是原论文完整复现。

## 结果与下一步

| 方法 | Best AP 均值 | Last AP 均值 | 平均训练秒数 |
| --- | ---: | ---: | ---: |
| cache1 | 2.5694 | 1.8258 | 213.6 |
| cache2 | 3.6405 | 2.8434 | 217.7 |
| FroFA C1，delta=0.1 | 2.7640 | 2.1806 | 218.0 |
| Transport C1 | 3.1775 | 2.7574 | 220.1 |
| LOFF-TA-style C1 | 3.2853 | 2.9942 | 223.3 |
| Transport C2 | 3.7666 | 3.4723 | 221.5 |
| LOFF-TA-style C2 | 3.7675 | 3.4479 | 232.2 |

全部强度和种子见 summary.csv、analysis.json 和 original/；包括表现较差的配置。

1. 当前训练协议下，C1 几何传输使平均 best AP 提高 0.6081，三个种子的差值均为正。旧的“Transport 接近随机”需核对当时的检测头、特征替换/重训练协议和实现，不能概括这一轮。
2. 双视角 Transport 作为下一阶段主要的便宜对照。相比 cache2，best AP 平均提高 0.1260，但不是每个种子都提高；last AP 平均提高 0.6288。
3. C2 加噪声只增加约 0.0009 mean best AP，last AP 略低，平均训练时间多约 4.8%。不能据此认定加噪声更优。
4. FroFA 弱增强在 C1 上有帮助，在 C2 上没有提高 mean best AP；这不等于否定原论文。

下一步是固定在线预算：C2-Mix、C2-Transport、C2-Correct，10% / 25% 骨干查询预算、种子 0/1/2，共 18 组。共享查询样本与增强参数、初始化和检测头；同一次骨干前向复用于检测训练及校正监督，完整计入开销。候选有效后扩大到正式划分、官方评估、小目标 AP、第二数据集和等 GPU 时间比较。本轮没有在线校正器结果。

## 证据与边界

已核对服务器 30 组各 100 行 CSV、轮次、有限值、实际配置和划分、汇总与逐轮 AP 一致性，以及 11 个源码指纹。这是产物一致性核验，没有重新从预测计算 AP 或独立复核全部标注。

original/reference_args.json 是早期继承模板，含旧 router_only 文本；真实执行设置见 original/train/*/args.json，30 组均核对为 mean_value。

本轮复用缓存，额外构建耗时为 0。build+wall 是历史训练缓存构建耗时加本轮运行时间，每组计一次，不按种子摊薄，排除已有验证缓存构建。当前比较均为 GPU0 串行运行。3 个种子、单一小样本划分和 best 验证成绩仅支持探索性结论。

服务器相关测试 11 passed，实验脚本关键错误检查通过。全仓 Ruff 有 2768 项既有问题，格式检查有 5 个既有文件；本次未修改这些已跟踪代码。codespell 未安装，未能执行。命令及退出码见 validation/，不宣称全仓检查通过。

已完成实验的训练源码原样提交，保留 SHA256 对应关系；不为格式整理改写旧源码或历史 provenance。

## 复现分析

在仓库根目录、带 matplotlib 的 Python 环境运行：

```bash
python online_experiments/analyze_feature_baselines.py research/experiments/feature-baselines-20260925-job-6JnMcLAf
```

- [分析报告](analysis-report.md)
- [统计细节](stats-appendix.md)
- [图表说明](figure-catalog.md)
- [原始证据索引](archive-manifest.json)

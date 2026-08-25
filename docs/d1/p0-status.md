# D1 P0 最低验收记录

## 结论

**PASS**

正式训练使用离线缓存的 DINOv3 多层特征，训练阶段未加载
DINOv3/teacher。可训练组件仅包括 P3/P4/P5 LatentMixture
与 Detect 检测头。

## 实验设置

- 数据集：COCO128 缓存样本
- 划分：80 train / 20 val
- 随机种子：0
- epochs：20
- 缓存输入：三层 384×40×40 FP16 特征
- 检测金字塔：P3=64×80×80、P4=128×40×40、P5=256×20×20
- 损失：真实 YOLO box/cls/dfl + LatentMixture auxiliary loss

## 最终结果

- precision：0.000684
- recall：0.001702
- mAP50：0.000136
- mAP50-95：0.000021
- 最佳 mAP50-95：0.000032
- 训练 GPU-hours：0.073546
- 峰值显存：1.520 GiB
- best.pt SHA256：`4ad97c5848996bb5705c4916330c7f346f4717cd102a184018a45d82ec1d6a42`
- last.pt SHA256：`7c59deb89d68f5b7ac8e580d6c471fdfffb86e4d6d776ea0153d077c312c4eee`

## 验收项

- 离线缓存读取：PASS
- 真实标签读取与 letterbox 坐标变换：PASS
- YOLO 检测损失前向/反向：PASS
- LatentMixture 辅助损失：PASS
- 验证集指标输出：PASS
- best/last checkpoint：PASS
- 教师模型未加载：PASS
- checkpoint 无教师参数：PASS

## 边界

本结果用于 P0 端到端链路验收。两个正式数据集、同预算基线、
多随机种子及节省比例属于 P1，不应使用本次 COCO128 小样本结果
替代 P1 结论。

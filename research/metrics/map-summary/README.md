# mAP 汇总与口径

本目录从已封存指标读取和计算类别平均，不重新训练、不推理，也不重新调用检测评估器。分数均为0–100。`map-results.csv`按数据集、方法、评估协议、maxDet分别列出结果；`sources.json`记录来源哈希和当前可用端点。

- `mAP50`：IoU=0.50时，各类别AP的等权平均。
- `mAP50_95`：IoU=0.50、0.55、…、0.95的十档AP，再对类别等权平均。不能只平均AP50和AP75得到。
- COCO/SKU的COCO bbox `AP`已执行上述类别和IoU平均，可标为`mAP50_95`；`AP50`可标为`mAP50`。SKU只有一个类别，类别平均不改变分数。maxDet100与500必须分列。
- VOC2007原生十一点AP50已经是20类等权`mAP50`。它不提供原生`mAP50_95`；该列留空。另有COCO-style辅助结果，不能混为VOC原生协议。原生VOC的maxDet500表示保存预测的全局上限，不表示另跑了maxDet扫描。
- VisDrone本项目参考实现的主AP按类别出现图像数加权，原值保留在`reference_weighted_AP`等列。`reference-class-macro`行使用同一参考匹配器、AP积分与预测，对十类结果等权平均；`COCO-style-auxiliary`行使用COCO匹配和101点插值。三者定义不同，不能只改列名后混用。

当前完整548图VisDrone参考评估器下的类别等权结果如下。它们不是追加实验，也不代表复现了YOLO-Master论文的训练配方。

| 方法 | 类别等权 mAP50–95 | 类别等权 mAP50 | 保留的参考加权AP |
| --- | ---: | ---: | ---: |
| YOLO-Master-N | 13.6234 | 24.2864 | 14.8749 |
| YOLO12-N | 13.2184 | 24.1644 | 14.3876 |
| FroFA思想适配 | 10.2285 | 21.7157 | 10.5272 |
| LOFF-TA思想适配 | 9.1352 | 19.3118 | 9.1234 |
| Cache and Reuse INT8适配 | 8.1118 | 17.5413 | 8.0720 |
| ViT-Adapter思想适配 | 15.8866 | 30.9682 | 16.5855 |
| Detail25 | 11.9406 | 24.9858 | 12.3453 |

Detail25另一个COCO-style辅助协议的mAP50–95@500=12.3109、mAP50@500=25.2895。论文表格必须说明使用哪套协议，不按分数高低选口径。旧真实Master的重复epoch日志审计继续保留；这里的分数换算不建立等训练步数或受控成本结论。

SKU已完成的真实YOLO-Master和YOLO12，在COCO原生maxDet100下，mAP50–95分别为40.3206、40.3693；maxDet500下分别为53.8890、54.1085。这就是先前表中的AP，无需再测一次。

后续端点完成后，刷新汇总只需读取已保存数值：

```bash
cd ~/YOLO-Master-paper-baselines-20261107
/mnt/data/users/zhjia/miniconda3/envs/yolomaster/bin/python \
  online_experiments/export_map_summary.py
```

训练和指标后台不受此命令影响；原始已封存指标文件保持不变。

# Statistical scope

The unit for method reliability is an independently trained model. Current n=1 training seed per method, with matched seed0 and full548 development images. Descriptive AP differences are reported in points, without inferential tests or intervals. Per-image distributions cannot replace training-seed replication. No observation was excluded for accuracy. Ignored GT/regions follow the fixed preprocessing; negative-size predictions remain zero-area FP candidates in auxiliary metrics.

COCO metric area ranges use inclusive endpoint conventions; exact1024/9216 area may enter neighboring groups. The descriptive GT-profile CSV instead uses disjoint intervals, explicitly documented in protocol.json.

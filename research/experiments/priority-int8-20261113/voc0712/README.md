# voc0712: CPU INT8 preparation

Existing Cache and Reuse detection adaptation; seed0/final100 training protocol unchanged. The original quantized_cache(job,8) implementation produces the physical bank. Original FP16 banks and manifests are unchanged. Later training reuses this bank; the quantizer's original build_seconds remains charged to cold cost. No image decoding, GPU lease, teacher call, training, or prediction is performed here. Only small manifests and CPU evidence are versioned; physical banks are not committed.

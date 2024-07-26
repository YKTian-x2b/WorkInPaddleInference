# WorkInPaddleInference
我在Paddle Inference的一些工作:

## LargeDIT模型推理优化
- [`README_largeDIT`](./LargeDIT/README_largeDIT.md)

## CUDA Kernel: MaskedMultiheadAttention
- 算子优化 seq_len维度切分，类似FlashDecoding
- [`README_MMHA`](./MaskedMultiheadAttention/README_MMHA.md)

## Cutlass生成GemmEpilogue算子
- 从pass到kernel
- [`README_GemmEpilogue`](./README_GemmEpilogue.md)

## Triton自定义算子：Norm相关
- 快速支持算子融合、模型优化
- [`README_triton`](./README_triton.md)


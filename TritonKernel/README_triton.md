# Triton Kernel

> https://github.com/PaddlePaddle/PaddleMIX/pull/552


- paddlemix/triton_ops/triton_ops.py


- 快速支持算子融合、模型优化
  - rmsNorm： 业务上+几ms
  - adaLN：largeDIT 几个点
  - fused_AdaLN：在adaLN基础上微弱提升
  - fused_rotary_emb： 2-4个点

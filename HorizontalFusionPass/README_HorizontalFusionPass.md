# HorizontalFusionPass

> https://github.com/PaddlePaddle/Paddle/pull/67004

- 支持MatmulOp/GemmEpilogueOp/FcOp的横向融合
- 类似于TensorRT的CBR
- 命中场景包括但不限于qkv的融合，ffn的融合，dit/sd中的多层里的matmul
- facebook/DiT-XL-2-256    0.7B 推理加速: 241.7 -> 234.7ms

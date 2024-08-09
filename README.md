# WorkInPaddleInference
我在Paddle Inference的一些工作:
- 2024.1.21-2024.8.12 一共提了11个pr，合入9个。


## LargeDIT模型推理优化
- TODO
- https://github.com/PaddlePaddle/PaddleMIX/pull/552
- LargeDIT 3B 最终耗时：581ms (提速 +61.2%)
- LargeDIT 7B 最终耗时：926ms (提速 +41.4%)
- [`README_largeDIT`](./LargeDIT/README_largeDIT.md)


## CUDA Kernel: MaskedMultiheadAttention
- https://github.com/PaddlePaddle/Paddle/pull/62838
- 算子优化 seq_len维度切分，类似FlashDecoding
- 加速效果见如下README：
- [`README_MMHA`](./MaskedMultiheadAttention/README_MMHA.md)


## Cutlass生成GemmEpilogue算子
- https://github.com/PaddlePaddle/Paddle/pull/61925
- 实现pass到kernel的整个流程
- facebook/DiT-XL-2-256 0.7B 推理加速: 481.21ms -> 311.779ms 
- [`README_GemmEpilogue`](./GemmEpilogue/README_GemmEpilogue.md)


## 横向融合pass
- https://github.com/PaddlePaddle/Paddle/pull/67004
- 支持MatmulOp/GemmEpilogueOp/FcOp的横向融合
- 命中场景包括但不限于qkv的融合，ffn的融合，facebook-dit中的多层里的matmul
- facebook/DiT-XL-2-256 0.7B 推理加速: 241.7ms -> 234.7ms
- [`README_HorizontalFusionPass`](./HorizontalFusionPass/README_HorizontalFusionPass.md)


## Triton自定义算子：Norm相关
- https://github.com/PaddlePaddle/PaddleMIX/pull/552
- 快速支持算子融合、模型优化
- rmsNorm、adaLN、fused_AdaLN、fused_rotary_emb
- [`README_triton`](./TritonKernel/README_triton.md)


## 其他
### CMake
[`CutlassCompileOpt`](./CutlassCompileOpt)


### 学习的东西
[`MyLearn`](./MyLearn)


~~~bash
/bin/cp -f ppdiffusers/examples/inference/class_conditional_image_generation-large_dit_3b.py /tyk/kaiPro/WorkInPaddleInference/LargeDIT
/bin/cp -f ppdiffusers/examples/inference/class_conditional_image_generation-large_dit_7b.py /tyk/kaiPro/WorkInPaddleInference/LargeDIT
/bin/cp -f ppdiffusers/ppdiffusers/models/dit_llama.py /tyk/kaiPro/WorkInPaddleInference/LargeDIT
/bin/cp -f ppdiffusers/ppdiffusers/models/modeling_utils.py /tyk/kaiPro/WorkInPaddleInference/LargeDIT
/bin/cp -f ppdiffusers/ppdiffusers/models/simplified_dit_llama.py /tyk/kaiPro/WorkInPaddleInference/LargeDIT
~~~
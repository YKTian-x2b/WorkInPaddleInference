# GemmEpilogue工作说明

> https://github.com/PaddlePaddle/Paddle/pull/61925

## 一、背景

我们的目标是要融合形如 **matmul + bias + act** 的模式，用cutlass编写op，生成多种内核配置，寻求更优的融合op实现。



## 二、现有的相关工作

- 一个是FcOp，用cublasLt实现的。我们的cutlass实现包含了FcOp的功能。
- 一个是onednn的实现
- 还有CPU端的实现



## 三、fuse_gemm现在支持的功能

- 加bias的时候（elementwiseAdd），除了常规全连接神经网络的 [M, N]+[1,N] 模式外，我们的**GemmEpilogueOp额外支持[M, N]+[M, N]的模**式。
  - 这是因为在跑大模型的时候，康康哥发现这种elementwiseAdd模式也是 常见且容易融合的。
  - [M,N] + [M,N]的意思是matmul的输出规模和bias的shape是一样的，对应元素加。
- 目前fc_fuse_pass支持 **cublasLt 和 cutlass 两种** 路径，后续会把选择路径的开关移到合理的位置（目前是个宏）。
- 因为cutlass原生支持了一些激活，所以GemmEpilogueOp也容易扩展激活，目前可以跑 relu和gelu。



## 四、融合op的性能

- 我们在llama和chatglm2上对比了散op和融合op的性能：

```cpp
python predictor.py --model_name_or_path ./inference/chatglm2-6b --dtype float16 \
--src_length 1024 --max_length 1024 --output_file ./inference/chatglm2-6b/output.json \
--decode_strategy greedy_search --mode static --batch_size 2
```

### chatglm2

- 28层的模型，共84个子图优化。
  - 子图种类0的MNK：**[48, 4608, 4096]**
    - 用的config：split_k==1 && <64, 64, 64>, <32, 32, 64>,<16, 8, 16>
  - 子图种类1的MNK：**[48, 4096, 4096]**
    - 用的config：split_k==1 && <64, 64, 64>, <32, 32, 64>,
  - 子图种类2的MNK：**[48, 4096, 13696]**
    - 用的config：split_k==1 && <64, 64, 64>, <32, 32, 64>,
  - 子图种类3的MNK：**[2, 4068, 4096]**
    - 用的config：split_k==1 && <16, 64, 64>,<16, 32, 64>,
  - 子图种类4的MNK：**[2, 4096, 4096]**
    - 用的config：split_k==1 && <16, 64, 64>,<16, 32, 64>,
  - 子图种类5的MNK：**[2, 4096, 13696]**
    - 用的config：split_k==1 && <64, 64, 64>, <32, 32, 64>,
    - 这里到底会选择<64,64,64>还是<16,64,64>（性能差不多），是不一定的。会受影响于：选配置的时候，该配置的运行位置。

| **单次运行**           | **融合前**  | **融合后**      | **加速比** |
| ---------------------- | ----------- | --------------- | ---------- |
| **均值**               | 4298.378 ms | **3942.706 ms** | **9.0%**   |
| **均值（去除波动值）** | 4094 ms     | **3773 ms**     | **8.5%**   |

- 修改workspace空间申请位置后：

| **连续十次运行均值** | **融合前**  | **融合后**        | **加速比** |
| -------------------- | ----------- | ----------------- | ---------- |
| **均值**             | 4298.378 ms | **3957.42908 ms** | **8.6%**   |

### llama

- 32层的模型，共64个子图优化。
  - 子图种类0的MNK：**[34, 4096, 4096]**
    - 用的config：split_k==1 && <64, 64, 64>, <32, 32, 64>,
  - 子图种类1的MNK：**[34, 4096, 11008]**
    - 用的config：split_k==1 && <64, 64, 64>, <32, 32, 64>,
  - 子图种类2的MNK：**[2, 4096, 4096]**
    - 用的config：split_k==1 && <16, 64, 64>,<16, 32, 64>,
  - 子图种类3的MNK：**[2, 4096, 11008]**
    - 用的config：split_k==1 && <64, 64, 64>, <32, 32, 64>,

| **5次实验\*连续十次运行取均值** | **融合前**     | **融合后**         | **加速比** |
| ------------------------------- | -------------- | ------------------ | ---------- |
| **均值**                        | 36165.52428 ms | **36138.59256 ms** | -          |
| **均值（去除波动值）**          | 36165.52428 ms | **35452.83295 ms** | **2.0%**   |

修改workspace空间申请位置后：

| **连续十次运行均值**   | **融合前**         | **融合后**    | **加速比** |
| ---------------------- | ------------------ | ------------- | ---------- |
| **均值**               | **36165.52428 ms** | 36720.2718 ms | -%         |
| **均值（去除波动值）** | **36165.52428 ms** | 36390.1107 ms | -%         |

swizzle后：

| **连续十次运行均值**   | **融合前**     |      **融合后**       | **加速比** |
| ---------------------- | -------------- | :-------------------: | ---------- |
| **均值**               | 36165.52428 ms |   **35428.1334 ms**   | **2.0 %**  |
| **均值（去除波动值）** | 36165.52428 ms | **34751.5052** **ms** | **4.0 %**  |



## 五、实现和调优

### 实现

- **内核实现来自cutlass/example**
  - 12_gemm_bias_relu/gemm_bias_relu.cu
  - 47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk.cu
  - 支持gemmUniversal和streamK
- **大部分配置参数直接来自cutlass的generator.py**
  - 如：threadblockshape/warpshape/mmashape/swizzle/num_stage
  - splitK范围是[1,2,4]。
- 因为**[2, 4096,\*]** 这种规模的问题，我们加入了 <16, 64, 64>,<16, 32, 64>,<16,8,16>的配置。

### 调优

- ***！！！因为个人水平的问题，cutlass实现可能还有很多调优空间未被发掘！！！***

#### split-k-factor的搜索空间

- 针对**[2, 4096,11008]** 问题未激活 splitK>1 配置的情况，我从AITemplate的cutlass调优部分，我找到split-k-factor的搜索范围如下：
  - 理论上splitK可选[1, 2]

```python
def _split_k_search_space(self, M, N, K):def _split_k_search_space(self, M, N, K):
        """Get split_k search range = [1] by default"""
        space = [1]
        # skip split-k search for rocm
        if backend.target.Target.current().name() == "rocm":
            return set(space)
        ### 这三行的意思 splitK应该在[1, 32]里。然后，splitK = [1/4, 1] * (K//max(M, N))
        ## 所以，在上述大模型里，也就是1或2
        factor = K // max(M, N)
        low_range = max(1, factor // 4)
        high_range = min(factor, 32)
        
        if low_range == 1:
            low_range += 1
        space += list(range(low_range, high_range, 2))
        _LOGGER.debug(
            f"profiling split-k for gemm instance M={M}, N={N}, K={K} in {set(space)}",
        )
        return set(space)
```

- splitK == 2 和 splitK == 1，在K规模提高的情况下，会有一个稳定的0.3ms左右的差距。splitK==1永远比splitK==2快0.3ms
  - tactic 0和1 是<64, 64, 64>, <32, 32, 64>,
  - tactic 9和10 是<16, 64, 64>,<16, 32, 64>,
  - 猜测差距来自：cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    - （ncu分析splitK已生效）。

```cpp
else if (mode == GemmUniversalMode::kGemm && grid_tiled_shape.k() > 1)
{
// Serial split-K only requires a temporary workspace if the number of partitions along the
// GEMM K dimension is greater than one.
workspace_bytes = sizeof(int) * size_t(grid_tiled_shape.m()) * size_t(grid_tiled_shape.n());
}。
we are tunning for problem: [2, 4096, 11008]
kai_____: fp16_fc_bias: tactic 0, cost_time: 0.764928ms.
kai_____: fp16_fc_bias: tactic 1, cost_time: 1.08237ms.
kai_____: fp16_fc_bias: tactic 9, cost_time: 0.709632ms.
kai_____: fp16_fc_bias: tactic 10, cost_time: 0.991232ms.

we are tunning for problem: [2, 4096, 44032]
kai_____: fp16_fc_bias: tactic 0, cost_time: 2.79962ms.
kai_____: fp16_fc_bias: tactic 1, cost_time: 3.10374ms.
kai_____: fp16_fc_bias: tactic 9, cost_time: 2.60608ms.
kai_____: fp16_fc_bias: tactic 10, cost_time: 2.84979ms.

we are tunning for problem: [2, 4096, 176128]
kai_____: fp16_fc_bias: tactic 0, cost_time: 10.9343ms.
kai_____: fp16_fc_bias: tactic 1, cost_time: 11.1585ms.
kai_____: fp16_fc_bias: tactic 9, cost_time: 10.2011ms.
kai_____: fp16_fc_bias: tactic 10, cost_time: 10.5595ms.
```

- 针对<16, 64, 64>,<16, 32, 64>,<16,8,16> & splitK=[1,2,4]的情况，分析了三种规模的输入
  - 猜测splitK在 **没跑满GPU且 K/max(M,N) 较大的情况下是有效的**

```html
we are tunning for problem: [2, 4096, 40000]
fc_bias_sm80_fp16_1 cost_time: 2.43098ms.
fc_bias_sm80_fp16_2 cost_time: 2.68083ms.
fc_bias_sm80_fp16_4 cost_time: 2.66342ms.

we are tunning for problem: [2, 1024, 40000]
fc_bias_sm80_fp16_1 cost_time: 1.74182ms.
fc_bias_sm80_fp16_2 cost_time: 1.08749ms.
fc_bias_sm80_fp16_4 cost_time: 0.869376ms.

we are tunning for problem: [2, 1024, 4000]
fc_bias_sm80_fp16_1 cost_time: 0.278528ms.
fc_bias_sm80_fp16_2 cost_time: 0.350208ms.
fc_bias_sm80_fp16_4 cost_time: 0.34304ms.
```



## 六、可能需要&还未支持

- 矩阵布局目前只支持RRR，其他Layout(如RCR)未实现，这意味着matmul的transpose参数被忽略了。
- 其他激活(如leakyRelu)未加入，待后续。
  - 各种激活所需的参数也就没有加入了。
- 为了不加入冗余，我放宽了FCInferMeta函数的约束以匹配额外模式。
  - 也就是说FcOp不能处理的模式，目前只在pass的约束中过滤，FCInferMeta中的check被取消了。



## 七、QA测试结果

- 大模型上均有加速

| model_name | batch_size | enable_trt | precision | enable_pir | avg_cost_last | avg_cost  | ***avg_cost_diff(%)*** | 90_cost_last | 90_cost   | 90_cost_diff(%) | cpu_mem_last | cpu_mem   | cpu_mem_diff(%) | gpu_mem_last | gpu_mem | gpu_mem_diff(%) |
| ---------- | ---------- | ---------- | --------- | ---------- | ------------- | --------- | ---------------------- | ------------ | --------- | --------------- | ------------ | --------- | --------------- | ------------ | ------- | --------------- |
| chatglm-6b | 1          | False      | fp16      | True       | 34371.284     | 33967.687 | **1.174**              | 34383.424    | 33990.442 | 1.143           | 13339.031    | 13419.664 | -0.604          | 12695        | 12703   | -0.063          |
| chatglm-6b | 8          | False      | fp16      | True       | 94907.294     | 94569.518 | 0.356                  | 94957.106    | 94591.522 | 0.385           | 13414.375    | 13364.547 | 0.371           | 15903        | 15967   | -0.402          |
| gpt-07b    | 1          | False      | fp16      | True       | 8415.076      | 7993.583  | **5.009**              | 8418.94      | 8003.846  | 4.93            | 2087.043     | 2118.758  | -1.52           | 24057        | 24059   | -0.008          |
| gpt-07b    | 8          | False      | fp16      | True       | 26137.455     | 25926.73  | 0.806                  | 26161.636    | 25948.198 | 0.816           | 2218.863     | 2128.981  | 4.051           | 24059        | 24059   | 0.0             |
| llama-7b   | 1          | False      | fp16      | True       | 35203.154     | 34678.455 | **1.49**               | 35249.415    | 34705.586 | 1.543           | 13900.938    | 13908.867 | -0.057          | 24059        | 24059   | 0.0             |
| llama-7b   | 4          | False      | fp16      | True       | 0.0           | nan       | nan                    | 0.0          | nan       | nan             | 13934.273    | 13877.031 | 0.411           | 24059        | 24059   | 0.0             |
| sd-1-5     | 1          | False      | fp16      | True       | 4577.966      | 4475.237  | **2.244**              | 4585.256     | 4477.703  | 2.346           | 4140.051     | 4128.836  | 0.271           | 4313         | 4353    | -0.927          |
| sdxl       | 1          | False      | fp16      | True       | 21702.598     | 20786.371 | **4.222**              | 21778.799    | 20856.631 | 4.234           | 11214.438    | 11158.629 | 0.498           | 14689        | 14691   | -0.014          |



## 八、DIT上的表现

- facebook/DiT-XL-2-256    0.7B

| 优化方法                                  | 性能记录      | 加速比     |
| ----------------------------------------- | ------------- | ---------- |
| **竞品：facebook/DiT-XL-2-256**           | **242ms**     |            |
| 动态图                                    | 1325.6738 ms  | 0.00%      |
| 装饰器动转静 + triton kernel + 细粒度算子 | 481.21ms      | 63.70%     |
| **exp_enable_use_cutlass**                | **311.779ms** | **76.48%** |

﻿

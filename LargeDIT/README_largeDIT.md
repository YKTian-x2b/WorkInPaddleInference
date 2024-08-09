# LargeDIT

> https://github.com/PaddlePaddle/PaddleMIX/pull/552



![img](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=b0bcce971d084e429a241bb93fc43f06&docGuid=bu0onWdXwvlMDL)

## 7B

| 优化方法         | 性能记录  | 加速比   |
| ---------------- | --------- | -------- |
| 动态图           | 1582 | 0.00%    |
| +装饰器动转静    | 1152      |          |
| +AdaLN+细粒度FFN | 1024 |          |
| +Fused_AdaLN     | 986       |          |
| +Fused_Repo      | 968       |          |
| 重新组网         | 926ms     | (+41.4%) |

## 3B

- 最终耗时 581ms(+61.2%)

﻿

﻿

## TODO

| TODO                                        | 时间/deadline | 责任人 | 备注                                                         |
| ------------------------------------------- | ------------- | ------ | ------------------------------------------------------------ |
| modulate+layerNorm => Triton kernel (AdaLN) | 已结束        | 田耀凯 | [Paddle_#64379](https://github.com/PaddlePaddle/Paddle/pull/64379)       pr是康康哥的代码 + 一点AdaLN |
| 结合PaddleNLP细粒度组网进行FFN细粒度组网    | 已结束        | 田耀凯 | [PaddleMIX_#552](https://github.com/PaddlePaddle/PaddleMIX/pull/552) |
| TransformerBlock 装饰器动转静               | 已结束        | 田耀凯 | TRT暂时搁置                                                  |
| Triton kernel: scale + residual +  AdaLN    | 已结束        | 田耀凯 | [Paddle_#64560](https://github.com/PaddlePaddle/Paddle/pull/64560) |
| qkv fineGrained                             | 已结束        | 田耀凯 | done                                                         |
| 整理pr中...                                 | 6.13前        | 田耀凯 | [PaddleMIX_#552](https://github.com/PaddlePaddle/PaddleMIX/pull/552) |
| **竞品调研**                                |               | 小田   |                                                              |
| **fused_bias_act 原生算子维度扩展**         | 已结束        | 小田   | [Paddle_#65302](https://github.com/PaddlePaddle/Paddle/pull/65302) |
| 剩余可融合散Op 转 Triton kernel             | Done          | 小田   | Silu Done.  Repo Done.                                       |
| 有效算子 + pass                             | Done          | 小田   | 横向融合算子                                                 |

﻿

## 优化工作

### AdaLN

- Triton算子 adaptive_layer_norm **融合LayerNorm + modulate**
- 动态图25个时间步的耗时测试：（Triton fp16的收益，可能得等没人用卡的时候，再看看，感觉结果有大波动）
  - 分辨率256*256

| 融合模式repeat次数_dtype | before   | after                      |
| ------------------------ | -------- | -------------------------- |
| 1_fp32                   | 1952.677 | **1925.083** **（+1.4%）** |
| 50_fp32                  | 6783.464 | **4036.333**               |

﻿

### fused_adaLN_scale_residual

- Triton算子 fused_adaLN_scale_residual **融合Scale + residual + LayerNorm + modulate**

﻿

### 细粒度FFN

- **FFN**用类似NLP LLama的**细粒度组网**
- 动态图25个时间步的耗时测试：
  - 分辨率256*256
  - 这里因为推理组网不是细粒度的，所以端到端不太能看，看only FFN比较合理。

| 融合模式repeat次数_dtype | before | after     |
| ------------------------ | ------ | --------- |
| 20_fp32                  | 23515  | **23384** |
| 20_fp16                  | 12940  | **11981** |

- 只记录FFN的收益：
  - 25时间步 * 32层 * 20次运行的均值

| 融合模式repeat次数_dtype | before | after              |
| ------------------------ | ------ | ------------------ |
| 20_fp32                  | 1.309  | 1.284**（+1.9%）** |
| 20_fp16                  | 0.692  | 0.632**（+8.6%）** |

﻿

### **修饰器动转静**

- **TransformerBlocks部分 修饰器动转静**
  - 没开TRT，只是32层transformerblock动转静。
- 静态图25个时间步的耗时测试：
  - 分辨率256*256

| 融合模式repeat次数_dtype | before | after               |
| ------------------------ | ------ | ------------------- |
| 1_fp32                   | 2017   | 1979**（+1.88%）**  |
| 1_fp16                   | 1648   | 1152**（+30.09%）** |

﻿

### 装饰器+AdaLN+细粒度FFN

- 静态图25个时间步的耗时测试：
  - 分辨率256*256

| 融合模式repeat次数_dtype | before   | after                   | after_sumSquare                                        |
| ------------------------ | -------- | ----------------------- | ------------------------------------------------------ |
| 1_fp32                   |          | **（+%）**              |                                                        |
| 1_fp16                   | 1864.937 | 1024.237**（+45.07%）** | total_time_cost: 1082.763mstotal_time_cost: 1081.789ms |

﻿

### **装饰器+Fused_AdaLN+AdaLN+细粒度FFN

- 静态图25个时间步的耗时测试：
  - 分辨率256*256

| 融合模式repeat次数_dtype | before   | after                   | after_606               | after+qkv_607      |
| ------------------------ | -------- | ----------------------- | ----------------------- | ------------------ |
| 1_fp32                   |          | **（+%）**              |                         |                    |
| 1_fp16                   | 1864.937 | 1017.005**（+45.46%）** | 1002.178**（+46.26%）** | 986**（+47.12%）** |

﻿

### Fused_AdaLN+AdaLN+细粒度FFN

- 动态图25个时间步的耗时测试：
  - 分辨率256*256

| 融合模式repeat次数_dtype | before   | after            |
| ------------------------ | -------- | ---------------- |
| 1_fp32                   |          | **（+%）**       |
| 1_fp16                   | 1864.937 | 1415**（+24%）** |

﻿

### 端到端

- 3B + 25步 + 256*256 + 新ir + 5次端到端取均值

|      | before   | after             | with repo_triton     |
| ---- | -------- | ----------------- | -------------------- |
| bf16 | 1498.995 | 647**（+56.8%）** | 594.98**（+60.3%）** |

- 7B + 25步 + 256*256 + 新ir + 5次端到端取均值

|      | before    | after             | with repo_triton     |
| ---- | --------- | ----------------- | -------------------- |
| bf16 | 1582.4206 | 993**（+37.2%）** | 968.12**（+38.8%）** |

﻿

﻿

## 启动

```python
# 指定预训练DiT权重参数位置
pipe = DiTPipeline.from_pretrained("Alpha-VLLM/Large-DiT-7B-256", paddle_dtype=dtype)
# 指定采样过程中 带噪latent样本的更新方法
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# 执行 num_inference_steps步 更新
image = pipe(class_labels=class_ids, num_inference_steps=25).images[0]
```

﻿

## 主要内容

- 采样的输入是随机的**latent:** z_T(B、C、sampleSize、sampleSize)、label类别和推理步数。

```python
latent_model_input = paddle.concat([latents] * 2) if guidance_scale > 1 else latents
class_labels_input = paddle.concat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels
self.scheduler.set_timesteps(num_inference_steps)
```

- pipe会进行 推理步数 的**循环**，在循环中执行 **DiT的前向传播****预测前一步的噪声，**然后根据噪声对带噪样本**x_t进行去噪操作得到x_t-1**。

```python
for t in self.progress_bar(self.scheduler.timesteps):
    # predict noise model_output （这里transformer == DiTLLaMA2DModel）
    noise_pred = self.transformer(latent_model_input, timestep=timesteps, class_labels=class_labels_input).sample
    # compute previous image: x_t -> x_t-1
    latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample
```

- 得到最终latent之后，将其输入**vae的解码器获得图片**。

```python
samples = self.vae.decode(latents).sample
```

﻿

## DiTLLaMA2DModel

![img](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=2b3aef9f43fb494ab3beb5a6a86f169f&docGuid=bu0onWdXwvlMDL)

### Input + DiT Block *  N + Output

- 如图左

```python
# 1. Input
hidden_states = self.patchify(hidden_states)
x = self.x_embedder(hidden_states)
t = self.t_embedder(timestep)
y = self.y_embedder(class_labels)
adaln_input = t + y
﻿
# 2. Blocks
for i, layer in enumerate(self.layers):
        x = layer(x, self.freqs_cis[: x.shape[1]], adaln_input,)
﻿
# 3. Output
hidden_states = self.final_layer(x, adaln_input)
output = self.unpatchify(hidden_states)
```

### DiT Block

- 如图右

```python
# alpha beta gamma 
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = 
                                self.adaLN_modulation(adaln_input).chunk(6, axis=1)
# LayerNorm + modulate + Attention + gate + residual
h = x + gate_msa.unsqueeze(1) * self.attention(
    modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis)
# LayerNorm + modulate + FFN + gate + residual
out = h + gate_mlp.unsqueeze(1) * self.feed_forward(modulate(self.ffn_norm(h), shift_mlp, scale_mlp))
```

﻿

### 主要耗时操作

![img](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=226a1736a5a348d5ab2f32b005c0f7e1&docGuid=bu0onWdXwvlMDL)

adaLN     wq           wk           wv                      attent   wo                 w1                                     w3                                       w2  

```python
# 1. gemmSN_NN (93us)
self.adaLN_modulation(adaln_input).chunk(6, axis=1)
﻿
# +LayerNorm (17us)
﻿
class Attention(nn.Layer):
    def forward(self, x, freqs_cis):
        # 2. cutlass::Kernel * 3 (160us、 161us、 160us)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 3. phi::fmha (123us)
        output, _ = flash_attention(xq,xk,xv,)
        # 4. cutlass::Kernel * 1 (161us)
        return self.wo(output)
﻿
# +LayerNorm (17us)
﻿
class FeedForward(nn.Layer):
    def forward(self, x):
        # 5. cutlass::Kernel * 3 (408us、 409us、 356us)
        xw1 = F.silu(self.w1(x))
        xw3 = self.w3(x)
        output = self.w2(xw1 * xw3)
```

﻿

### 耗时分析

- 一个TransformerBlock的总耗时：2394us
- 主要是：gemmSN_NN (94us) + LayerNorm (18us) + cutlass::Kernel * 3 (160us、 160us、 160us) + fmha (123us) + cutlass::Kernel * 1 (160us) + cutlass::Kernel * 3 (408us、 408us、 355us)
- 如下是所有耗时细节：

| API                                                          | 算子                                      | 耗时       | 占比        | **融合思路**                              | 备注                |
| ------------------------------------------------------------ | ----------------------------------------- | ---------- | ----------- | ----------------------------------------- | ------------------- |
| **self.adaLN_modulation**                                    | **gemmSN_NN**                             | 94us       | **3.926%**  | Linear                                    | **-**               |
| **nn.LayerNorm**                                             | **LayerNorm**                             | 18us       | **0.752%**  | **AdaLN_Triton**                          | **已完成**          |
| modulate(scale + shift)                                      | broadcast*2                               | 22us       | **0.919%**  |                                           |                     |
| **xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)**          | **cutlass::Kernel \* 3**                  | 480us      | **20.050%** | **cutlass::Kernel \* 1**                  | **细粒度**          |
| self.q_norm(xq)/self.k_norm(xk) / xq.reshape/xk.reshape/xv.reshape / Attention.apply_rotary_emb(xq, xk) | fast_ln_fwd*2 + (elementwise+broadcast)*n | 108us      | **4.511%**  | 散Op                                      | 没细看              |
| **scaled_dot_product_attention_**                            | **phi::fmha**                             | 123us      | **5.138%**  | fmha                                      | **-**               |
| **self.wo(output)**                                          | **cutlass::Kernel**                       | 160us      | **6.683%**  | Linear                                    | **-**               |
| scale + residual                                             | broadcast*2                               | 23us       | **0.961%**  | **Fused_AdaLN_Scale_Residual ？**         | **已完成**          |
| **nn.LayerNorm**                                             | **LayerNorm**                             | 17us       | **0.710%**  |                                           |                     |
| modulate(scale + shift)                                      | broadcast*2                               | 24us       | **1.003%**  |                                           |                     |
| **xw1_tmp = self.w1(x)**                                     | **cutlass::Kernel**                       | 408us      | **17.043%** | **SwiGLU**                                | **细粒度 & 已完成** |
| xw1 = F.silu(xw1_tmp)                                        | elementwise                               | 25us       | **1.044%**  |                                           |                     |
| **xw3 = self.w3(x)**                                         | **cutlass::Kernel**                       | 408us      | **17.043%** |                                           |                     |
| **xw13_tmp = xw1 \* xw3**                                    | **broadcast**                             | 43us       | **1.796%**  |                                           |                     |
| **output = self.w2(xw13_tmp)**                               | **cutlass::Kernel**                       | 355us      | **14.829%** | Linear                                    | **-**               |
| scale + residual                                             | broadcast*2                               | 20us       | **0.835%**  | 可以和下一层Fused_AdaLN_Scale_Residual ？ | 组网不好改/暂时不看 |
| **汇总**                                                     |                                           | **2328us** | **97.243%** | -                                         |                     |

![img](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=49be185f5535400a90f2f81e99f55ce3&docGuid=bu0onWdXwvlMDL)

- 耗时1.5%以上的 拉个表 容易看。

| API                                                          | 算子                                      | 耗时        | 占比        | 备注           |
| ------------------------------------------------------------ | ----------------------------------------- | ----------- | ----------- | -------------- |
| **self.adaLN_modulation**                                    | **gemmSN_NN**                             | 94us        | **3.926%**  | Attention ~40% |
| **xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)**          | **cutlass::Kernel \* 3**                  | 160*3=480us | **20.050%** |                |
| self.q_norm(xq)/self.k_norm(xk) / xq.reshape/xk.reshape/xv.reshape / Attention.apply_rotary_emb(xq, xk) | fast_ln_fwd*2 + (elementwise+broadcast)*n | 108us       | **4.511%**  |                |
| **scaled_dot_product_attention_**                            | **phi::fmha**                             | 123us       | **5.138%**  |                |
| **self.wo(output)**                                          | **cutlass::Kernel**                       | 160us       | **6.683%**  |                |
| **xw1_tmp = self.w1(x)**                                     | **cutlass::Kernel**                       | 408us       | **17.043%** | FFN ~50%       |
| **xw3 = self.w3(x)**                                         | **cutlass::Kernel**                       | 408us       | **17.043%** |                |
| **xw13_tmp = xw1 \* xw3**                                    | **broadcast**                             | 43us        | **1.796%**  |                |
| **output = self.w2(xw13_tmp)**                               | **cutlass::Kernel**                       | 355us       | **14.829%** |                |
| **汇总**                                                     |                                           | **2179us**  | **91.019%** |                |

﻿

## 数据流动

```python
self.sample_size 32
self.patch_size 2
self.in_channels 4
self.out_channels 8
self.num_layers 32
self.num_attention_heads 32
self.mlp_ratio 4.0
self.multiple_of 256
self.ffn_dim_multiplier None
self.norm_eps 1e-05
self.class_dropout_prob 0.1
self.num_classes 1000
self.learn_sigma True
self.qk_norm True
dim 4096
```

##### 输入

- 输入**latent**(N, C_in, sampleSize, sampleSize) -> **patchify**(N**,** sampleSize//patchSize * sampleSize//patchSize**,** C_in*patchSize*patchSize)
- **latentEmbedder**(N**,** sampleSize//patchSize * sampleSize//patchSize**,** dim); **TimestepEmbedder**(N, dim); **LabelEmbedding**(N, dim); 

##### DiT

- (shift, scale, gate)(_mlp/_msa)=**adaLN_modulation**(N, dim)
- **LayerNorm + modulate** =》 (N**,** sampleSize//patchSize * sampleSize//patchSize**,** dim)
- **Attention** =》(N**,** sampleSize//patchSize * sampleSize//patchSize**,** dim)
- **gate + residua**l =》 (N**,** sampleSize//patchSize * sampleSize//patchSize**,** dim)
- **LayerNorm + modulate** =》 (N**,** sampleSize//patchSize * sampleSize//patchSize**,** dim)
- **FFN** =》(N**,** sampleSize//patchSize * sampleSize//patchSize**,** dim)
- **gate + residual** =》 (N**,** sampleSize//patchSize * sampleSize//patchSize**,** dim)

##### 输出

- (shift, scale) = **adaLN_modulation**(N, hidden_dim)
- **LayerNorm + modulate** =》 (N**,** sampleSize//patchSize * sampleSize//patchSize**,** T**,** hidden_dim)
- **Linear** =》(N**,** sampleSize//patchSize * sampleSize//patchSize**,** C_out*patchSize*patchSize)
- **unpatchify**(N, C_out, sampleSize, sampleSize)

﻿

##### VAE

- ﻿

﻿

﻿

## 其他内容

- DiT 采用了 Classifier-free guidance（CFG）机制，使用引导系数 guidance_scale 混合条件和无条件噪声作为预测噪声

![img](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=f7207c556bd14fa18f97d8e35d65c4f1&docGuid=bu0onWdXwvlMDL)

```python
if guidance_scale > 1:
    eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
    cond_eps, uncond_eps = paddle.chunk(eps, 2, axis=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = paddle.concat([half_eps, half_eps], axis=0)
    noise_pred = paddle.concat([eps, rest], axis=1)
```

- DDIM样本更新：

![img](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=ce4cf63db6da4fd48dad165b53757a99&docGuid=bu0onWdXwvlMDL)

```python
# 3. compute predicted original sample from predicted noise also called
# "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
# 5. compute variance: "sigma_t(η)" -> see formula (16)
# σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
variance = self._get_variance(timestep, prev_timestep)
std_dev_t = eta * variance ** (0.5)
# 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
# 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
#
prev_sample = prev_sample + variance
```

﻿

﻿
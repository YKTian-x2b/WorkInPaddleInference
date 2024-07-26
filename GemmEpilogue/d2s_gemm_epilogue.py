import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse
import paddle 
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType

from paddle import nn
from paddle.static import InputSpec


class gemm_epilogue(nn.Layer):
  def __init__(self, K):
    super().__init__()
    self.K = K
    self.linear = paddle.nn.Linear(self.K, self.K)
    self.linear.weight.to('float16')
    self.linear.bias.to('float16')

  def forward(self, x, weight, bias):
    x_ = self.linear(x)
    matmul_out = paddle.matmul(x, weight)
    # out = paddle.add(matmul_out, bias) 
    out = paddle.add(bias, matmul_out) 
    return out, x_


def init_predictor():
    model_dir = "/tyk/Paddle/kai/triton/profile"
    model_file = model_dir + "/gemm_epilogue_fp16.pdmodel"
    param_file = model_dir + "/gemm_epilogue_fp16.pdiparams"
    gpu_precision = PrecisionType.Half

    config = Config(model_file, param_file)

    config.enable_new_ir()
    config.enable_memory_optim()
    config.enable_use_gpu(100, 0, gpu_precision)
    config.exp_enable_use_cutlass()

    predictor = create_predictor(config)
    return predictor

def dy2st(x, weight, bias, K):
    ############ 动转静 ############
    x_spec = InputSpec.from_tensor(x, name='x')
    weight_spec = InputSpec.from_tensor(weight, name='weight')
    bias_spec = InputSpec.from_tensor(bias, name='bias')

    gemm_epilogue_layer = gemm_epilogue(K)
    dy2st_gemm_epilogue_layer = paddle.jit.to_static(gemm_epilogue_layer, 
                                                        input_spec=[x_spec, weight_spec, bias_spec])

    path = "/tyk/Paddle/kai/triton/profile/gemm_epilogue_fp16"
    paddle.jit.save(dy2st_gemm_epilogue_layer, path)


def inference(x, weight, bias):
    ############ 加载模型并推理 ############
    pred = init_predictor()
    d2s_out, _ = pred.run([x, weight, bias])  
    print(d2s_out.shape)

    print("\n---------- valid ------------")
    paddle.disable_static()
    out = paddle.add(paddle.matmul(x, weight), bias) 
    print(d2s_out.dtype, out.dtype)
    print(out[0,0,:])
    print(d2s_out[0,0,:])
    print("maxdiff: ", paddle.max(paddle.abs(d2s_out - out)))
    res = paddle.allclose(d2s_out, out, rtol=0., atol=1e-02)
    print("allclose: ", res) 


if __name__ == "__main__":
    dtype='float16'

    batch_size = 8    
    seq = 64
    K = 1024
    N = 6*1024    #  6625
    
    x = paddle.rand([batch_size, seq, K], dtype=dtype)
    weight = paddle.rand([K, N], dtype=dtype)
    # bias = paddle.rand([N], dtype=dtype)
    bias = paddle.rand([batch_size, seq, N], dtype=dtype)

    dy2st(x, weight, bias, K) 
    inference(x, weight, bias)
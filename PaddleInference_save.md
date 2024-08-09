# Inference

## 可重用脚本

#### Paddle脚本

```bash
ssh tianyaokai@relay.baidu-int.com
ssh tianyaokai@yq01-inf-hic-k8s-a100-aa24-0087.yq01.baidu.com
cd /ssd2/tyk

# paddle编译
# 需要修改 cmake/cuda11.6/python/pip/protobuf/patchelf
# 运行单元测试需要 -DWITH_TESTING=ON
cmake .. -DPY_VERSION=3.8     -DWITH_TESTING=OFF   -DWITH_MKL=ON     -DWITH_GPU=ON     -DON_INFER=ON     -DWITH_TENSORRT=ON -DTENSORRT_ROOT=/usr/local/tensorrt/ -DWITH_INFERENCE_API_TEST=OFF -DWITH_DISTRIBUTE=OFF -DEXP_CUDA_MODULE_LOADING_LAZY=ON -DWITH_INFERENCE_NVTX=ON -DCMAKE_CXX_FLAGS='-Wno-error=unknown-pragmas' -DWITH_NVTX=ON
# with trt
cmake .. -DPY_VERSION=3.8         -DWITH_TESTING=OFF     -DWITH_MKL=ON         -DWITH_GPU=ON         -DON_INFER=ON         -DWITH_TENSORRT=ON -DTENSORRT_ROOT=/usr/local/tensorrt/ -DWITH_INFERENCE_API_TEST=ON -DWITH_DISTRIBUTE=ON -DEXP_CUDA_MODULE_LOADING_LAZY=ON -DWITH_INFERENCE_NVTX=OFF

# 新容器
cmake .. -DPY_VERSION=3.10     -DWITH_TESTING=OFF   -DWITH_MKL=ON     -DWITH_GPU=ON     -DON_INFER=ON     -DWITH_TENSORRT=ON -DTENSORRT_ROOT=/usr/local/tensorrt/ -DWITH_INFERENCE_API_TEST=OFF -DWITH_DISTRIBUTE=OFF -DEXP_CUDA_MODULE_LOADING_LAZY=ON -DWITH_INFERENCE_NVTX=ON -DCMAKE_CXX_FLAGS='-Wno-error=unknown-pragmas' -DWITH_NVTX=ON

# 编译paddle
cd /tyk/Paddle/build && make -j 40
cd /tyk/Paddle/build/python/dist && python3 -m pip uninstall -y paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
cd /tyk/Paddle/build/python/dist && python3 -m pip install -U paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl

cd /tyk/Paddle/build && make -j 40 && cd /tyk/Paddle/build/python/dist && python3 -m pip uninstall -y paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl && python3 -m pip install -U paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl

# 清除A100空间
rm -rf /root/.cache
rm -rf /root/.ccache
# 服务器到主机文件传输
abc=`hostname -i` && port=8484 && echo $abc:$port && python -m http.server $port


# icoding: https://icoding.baidu-int.com/workspace/
curl -s http://baidu-ide.bj.bcebos.com/platform/script/host-script/install-agent.sh | bash -s -- -g 47187e62-b15c-4bce-93b1-f2c910b41cb3 -c ca7710e48b33f4fa201dfc60d53d088e -v 1.8.401.83.1.02 -p 40000
curl -s http://baidu-ide.bj.bcebos.com/platform/script/host-script/install-agent.sh | bash -s -- -g 1b8514f1-baae-4775-999d-6c7db98a42e0 -c 7184998d1b87a202e9250c3b2211ecdc -v 1.8.401.83.1.02 -p 10240	
```



#### Docker脚本

```bash
# 启动cuda11.6容器
sudo nvidia-docker run --name zkk-work-first -v $PWD:/zhoukangkang --network=host -it cd7ed4c5f55f /bin/bash
# 为了用nsight compute 启动的时候要加上 --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt
sudo nvidia-docker run --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --name zkk-work-nc -v $PWD:/zhoukangkang --network=host -it 6594a7a243f0  /bin/bash
# for me
nvidia-docker run --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --name tyk-work-ncu -v $PWD:/tyk --network=host -it cd7ed4c5f55f /bin/bash
# 连接docker
sudo nvidia-docker exec -it zkk-work-nc /bin/bash

#### docker命令
docker exec -it tyk-work-ncu bash  # 非绑定式进入容器
docker ps  # 查看正在运行的容器
docker images   # 查看镜像
# 重新生成一个镜像（生成过程要出docker）；容器id，起个name：
# docker commit 79cb8086fe36 tyk-paddle-image
docker commit id name
# 保存为本地文件
# docker save -o tyk-paddle-imagefile.tar tyk-paddle-image
docker save -o  文件名   镜像名
# 加载镜像
# docker load -i tyk-paddle-imagefile.tar
docker load -i  文件名   镜像名


#### 系统命令
wget http://10.78.119.13:8088/cuda-11.6.tar		# cuda11.6 软链接到usr/bin/cuda
lsb_release -a  # 查看系统类型
```



#### 大模型脚本

```bash
# 大模型
python3 -m pip install -r requirements.txt
python3.8 setup.py develop
# nsys profile -t nvtx,osrt --stats=true -f true -o BloomSysProf
python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 1024 --batch_size 1 --inference > /tyk/PaddleNLP/llm/kai/res_after_1024.txt	# res_before
# 编paddle的时候，-DWITH_NVTX=ON，然后config.enable_profile()，然后执行脚本的时候nsys profile -t nvtx,cuda -o profile.log -f true ...

cd PaddleNLP/llm
# facebook/llama-7b  # THUDM/chatglm2-6b   #baichuan-inc/Baichuan2-7B-Base	# THUDM/chatglm-6b
# meta-llama/Llama-2-7b-chat
python export_model.py --model_name_or_path meta-llama/Llama-2-7b-chat --output_path ./inference/Llama-2-7b-chat  --dtype float16 
python export_model.py --model_name_or_path facebook/llama-7b --output_path ./inference/llama-7b_715 --dtype float16
# nsys profile -t nvtx,osrt --stats=true -f true -o /tyk/PaddleNLP/llm/kai/chatglm2SysProf 
python predictor.py --model_name_or_path ./inference/llama --dtype float16 --src_length 1024 --max_length 1024 --output_file ./inference/llama/output.json --decode_strategy greedy_search --mode static --batch_size 2


# TrtLLM
python3.10 -m venv --without-pip trtllm_venv
# --clean --cpp_only 
python3.10 ./scripts/build_wheel.py --clean --trt_root /usr/local/tensorrt --cuda_architectures "80-real;86-real"
```



#### Git脚本

```bash
# 如果只是小改动 比如没改cmakelists什么的 可以尝试
rm CMakeCache.txt # 然后再重新cmake
# 重新merge build报错的时候 需要重新cmake 甚至需要
git clean -xfd # 然后删掉build目录。warning：这个行为会把没有git记录的所有文件删掉！！！

# 如何新增pass
https://github.com/PaddlePaddle/Paddle/pull/58680

## 怎么提pr
https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/git_guides/local_dev_guide_cn.html
# 首先fork要pr的源仓库到自己仓库，然后在develop分支上branch一个分支 用作自己修改
git checkout -b my-cool-stuff
# 改完之后 add commit，push前同步源仓库
git remote add upstream https://github.com/PaddlePaddle/Paddle
git fetch upstream
git pull upstream develop
# 推送到自己仓库origin的my-cool-stuff分支上
git push origin my-cool-stuff

# 拉取未合入的pr
git fetch upstream pull/63333/head:jieru_triton
git checkout jieru_triton

# 直接把已经跟踪的文件，被你改动过的，直接commit啦！
git commit -a -m "first commit"
#  --no-verify 跳过pre-commit检查

# 直接看别的分支的文件
git show my-cool-stuff:/tyk/Paddle/paddle/phi/build/generated_tmp/matmul_add_sm80_fp16_189.cu
```



#### NV脚本

```bash
# nvtx过滤分析内容为“loop”区间，应用重放，全指标分析，强制覆盖写profile文件，应用为python执行
cd /tyk/Paddle/kai && ncu --nvtx --nvtx-include "fmha" --nvtx-exclude "beforeLoop" --replay-mode=application --set full -f -o profile python 01-many-mmha.py 
# 默认是kernel，还有range
--replay-mode=application
# nsys脚本
nsys profile --stats=true -f true -o splitK ./splitK
nsys profile -t nvtx,osrt --stats=true -f true -o syspf python 01-many-mmha.py	# -t只看nvtx,osrt
# ncu脚本
ncu -f -o profile --set full python 01-many-mmha.py
# 编译kernel时需要加上-lineinfo flag，这样在report中才能看到sass对应的source，以sm80为例
# nvcc --generate-line-info
nvcc -arch=sm_80 -lineinfo xxx.cu

# 锁频
nvidia-smi -i 0 -pm	1			# -i指定GPU -pm开启持久模式 减少GPU软硬件的初始化和取消初始化次数
-lgc -lmc									# 锁gpu和memory频率

# nsys 过滤？
nsys -t cuda,nvtx --capture-range=nvtx --nvtx-capture='loop' --nvtx-domain-include=default
```



#### 性能测试脚本

- warmup
- 被测试部分可以repeat多次来放大性能差距
- 前后要进行设备同步
- 端到端又可以多次求均值

```python
# kai mod warmup
self._infer(tokenized_source)

from datetime import datetime
paddle.device.cuda.synchronize(0)
starttime = datetime.now()
_infer_nvtx = nvtx.start_range(message="_infer", color="blue", domain='_inferDomain')

for kk in range(5):
		predictions = self._infer(tokenized_source)

paddle.device.cuda.synchronize(0)
nvtx.end_range(_infer_nvtx)
endtime = datetime.now()
duringtime = endtime-starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
msg = "total costs: " + str(time_ms/5) + "ms\n\n\n"
print(msg)
with open("/tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_715/time_bestR_1024_4_716_none_1237.txt", "a") as time_file:
    time_file.write(msg)
```



#### Python脚本

```bash
# 看看包下载在哪了
python -m site
pip show triton
```



#### 代理

~~~bash
# 代理
export http_proxy=http://172.19.56.199:3128
export https_proxy=http://172.19.56.199:3128
export no_proxy=bcebos.com
# [也可考虑另一代理，网速更快]
export https_proxy=http://172.19.57.45:3128
export http_proxy=http://172.19.57.45:3128
export GIT_SSL_NO_VERIFY=1
# [备选代理1]
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,pypi.tuna.tsinghua.edu.cn,paddle-ci.gz.bcebos.com && export http_proxy=http://10.162.37.16:8128 && export https_proxy=http://10.162.37.16:8128
# [备选代理2]
export http_proxy=agent.baidu.com:8118 && export https_proxy=agent.baidu.com:8118
#
export http_proxy=http://172.19.56.199:3128
export https_proxy=http://172.19.56.199:3128
export no_proxy=bcebos.com,pip.baidu-int.com,mirrors.baidubce.com,repo.baidubce.com,repo.bcm.baidubce.com,pypi.baidu.com
# git 代理
git config --global http.proxy agent.baidu.com:8118
git config --global https.proxy agent.baidu.com:8118
~~~



## 暂存

- https://github.com/PaddlePaddle/Paddle/pull/58680 这个pr完整的添加了一个融合算子，相应的pir pass以及对应的单测，可以参考下

- https://github.com/PaddlePaddle/Paddle/pull/62901#pullrequestreview-1959611460 @田耀凯  onednn他们写的这个pass pattern和单测的非常细致，咱们如果cutlass的算子也能搞成这样，就非常好了！

- https://docs.nvidia.com/cuda/cublasdx/introduction1.html#defining-gemm-operation 给大家分享一个cublasdx，基于thread-block-wide 的 matrix multiplication API可以比较方便实现EPILOGUE和PROLOGUE的集成，以便实现诸如mha、gemm_with_dequant等等操作。 @田耀凯 耀凯关注呢

- 装饰器

  ~~~python
  @paddle.incubate.jit.inference(enable_new_ir=True,
                                cache_static_model=True,
                                exp_enable_use_cutlass=True,
                                delete_pass_lists = ["trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass",
                                      "add_support_int8_pass", 
                                      "fc_fuse_pass", 
                                      "add_norm_fuse_pass",])
  ~~~

- export TRITON_KERNEL_CACHE_DIR=/root/.cache




## 学到的

### CUDA

- L2 miss的DRAM读取粒度为64字节，即L2 cacheline的下半部分或上半部分。即L2 cacheline的是128字节的，但读取粒度可以不是。
- Misaligned address错误往往发生在自行推导指针的时候。至于内置的类型，不需要指定align属性的（因为都自带）。未见使用CUDA内置类型，而不是自行使用指针推导的时候出现这个情况的可能。
  - 思考这个错误，要看warp整体访存的粒度是不是比存储块大。
- 深度学习框架中，数据一般是4D的，NCHW或NHWC。前者先存完图片的R，再存G，最后存B；后者连续存储一个像素点的R，G和B。
  - N for Batch
  - C for Channel
  - H for Height
  - W for Width
- BlockSwizzle：N越大 logtile越大，右矩阵读的轮数越少。N==1的时候，就没有swizzle。
- 前沿模型算子库：
  - https://github.com/facebookresearch/xformers
  - https://github.com/facebookincubator/AITemplate
  - https://github.com/NVIDIA/FasterTransformer
  - https://github.com/NVIDIA/TensorRT-LLM
  - OpenDIT OpenSora OneDiff
  - https://github.com/Dao-AILab/flash-attention
- mixtral moe

### C++

- 变量传递的每个环节都要考虑是否可以加 const &；
- 类的构造函数是否需要implicit
- 代码要尽可能精简

### Cutlass split-k Tune

~~~C++
def _split_k_search_space(self, M, N, K):
        """Get split_k search range = [1] by default"""
        space = [1]
        # skip split-k search for rocm
        if backend.target.Target.current().name() == "rocm":
            return set(space)
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
~~~




## 成果

- cutlass通用fc算子生成 https://github.com/PaddlePaddle/Paddle/pull/61925
- DiT 细粒度优化 https://github.com/PaddlePaddle/PaddleMIX/pull/552
- triton adaLN算子实现 https://github.com/PaddlePaddle/Paddle/pull/64379
- MMHA算子优化  [#62838](https://github.com/PaddlePaddle/Paddle/pull/62838)

  - flashattention2/pagedattention/smoothQuant
- Pass怎么写 https://github.com/PaddlePaddle/Paddle/pull/67004
- nsight系列分析大型项目的某块算子性能
- docker/cmake/c++

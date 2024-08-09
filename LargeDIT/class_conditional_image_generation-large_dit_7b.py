# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import paddle
from paddlenlp.trainer import set_seed

from ppdiffusers import DDIMScheduler, DiTPipeline

dtype = paddle.bfloat16

# True for inference optimizate
os.environ['INFERENCE_OPTIMIZE'] = "True"

with paddle.LazyGuard():
    pipe = DiTPipeline.from_pretrained("Alpha-VLLM/Large-DiT-7B-256", paddle_dtype=dtype)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
set_seed(0)

words = ["golden retriever"]  # class_ids [207]
class_ids = pipe.get_label_ids(words)

image = pipe(class_labels=class_ids, num_inference_steps=25).images[0]

warm_up_time = 2
repeat_time = 5
import datetime

for i in range(warm_up_time):
    image = pipe(class_labels=class_ids, num_inference_steps=25).images[0]

paddle.device.cuda.synchronize(0)
starttime = datetime.datetime.now()

for i in range(repeat_time):
    image = pipe(class_labels=class_ids, num_inference_steps=25).images[0]

paddle.device.cuda.synchronize(0)
endtime = datetime.datetime.now()
duringtime = endtime-starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
msg = "total costs: " + str(time_ms/repeat_time) + "ms\n\n\n"
print(msg)
with open("/tyk/PaddleMIX/ppdiffusers/examples/inference/kai/res/res_808/7B_time.txt", "a") as time_file:
    time_file.write(msg)

image.save("class_conditional_image_generation-large_dit_7b-result.png")

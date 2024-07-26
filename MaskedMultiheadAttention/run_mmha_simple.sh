set -e


echo "==================================================origin_717"

cd /tyk/Paddle/kai/mmha/paddleWhl

/bin/cp -f paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64_origin_717.whl paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl

pip3.8 uninstall --yes paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl 

pip3.8 install -U paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl

cd /tyk/PaddleNLP/llm

python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 1024 \
                       --batch_size 1 --benchmark --inference_model > /tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_716/res_before_1024_1__dsMIX_attn_718.txt 

python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 1024 \
                       --batch_size 2 --benchmark --inference_model > /tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_716/res_before_1024_2__dsMIX_attn_718.txt 

python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 1024 \
                       --batch_size 4 --benchmark --inference_model > /tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_716/res_before_1024_4__dsMIX_attn_718.txt 

python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 1024 \
                       --batch_size 8 --benchmark --inference_model > /tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_716/res_before_1024_8__dsMIX_attn_718.txt 

python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 2048 \
                       --batch_size 1 --benchmark --inference_model > /tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_716/res_before_2048_1__dsMIX_attn_718.txt 

python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 2048 \
                       --batch_size 2 --benchmark --inference_model > /tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_716/res_before_2048_2__dsMIX_attn_718.txt 

python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 3072 \
                       --batch_size 1 --benchmark --inference_model > /tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_716/res_before_3072_1__dsMIX_attn_718.txt 




echo "==================================================after_716pr_717"

cd /tyk/Paddle/kai/mmha/paddleWhl

/bin/cp -f paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64_716pr_717.whl paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl

pip3.8 uninstall --yes paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl

pip3.8 install -U paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl

cd /tyk/PaddleNLP/llm

python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 1024 \
                       --batch_size 1 --benchmark --inference_model > /tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_716/res_after_716pr_1024_1__dsMIX_attn_718.txt 

python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 1024 \
                       --batch_size 2 --benchmark --inference_model > /tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_716/res_after_716pr_1024_2__dsMIX_attn_718.txt 

python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 1024 \
                       --batch_size 4 --benchmark --inference_model > /tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_716/res_after_716pr_1024_4__dsMIX_attn_718.txt 

python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 1024 \
                       --batch_size 8 --benchmark --inference_model > /tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_716/res_after_716pr_1024_8__dsMIX_attn_718.txt 

python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 2048 \
                       --batch_size 1 --benchmark --inference_model > /tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_716/res_after_716pr_2048_1__dsMIX_attn_718.txt 

python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 2048 \
                       --batch_size 2 --benchmark --inference_model > /tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_716/res_after_716pr_2048_2__dsMIX_attn_718.txt 

python3.8 predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --dtype float16 --src_length 1024 --max_length 3072 \
                       --batch_size 1 --benchmark --inference_model > /tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_716/res_after_716pr_3072_1__dsMIX_attn_718.txt 
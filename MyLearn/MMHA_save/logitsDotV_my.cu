/// kai mod: 重写 logits dot cache_v 
  /// 思路就是仿照cache_k 存算 cache_v。不同的是最小的Dh是32，我们希望一个block的8个warp分掉这个32。1个warp就是4个Dh元素.
  /// 4个线程要读128B数据，就要把4个Dh元素扩充成128B。如果一个qk_vec占用了两个Dh元素，那么就肯定有线程要闲置。比起warp内分支，更倾向于使用半数threadblock。
  /// 这样做需要零填充最后一个seq的前几行，让它可以横向折叠。零填充的执行位置待定。
  /// kai_cache_v [batch, num_head, max_seq_len/v_vecs, head_dim/qk_vec_size, qk_vec_size*v_vecs]  32B == qk_vec_size*v_vecs*sizeof(T)
  // 读cache_v 一个warp的同排4个线程应该读128B的全局内存，一个线程32B
  constexpr int V_ELTS_IN_32B = 32 / sizeof(T);
  // 一个32B需要多少个Qk_vec（线程）来填充
  constexpr int V_VECS_IN_32B = 32 / sizeof(Qk_vec);
  // 指向kai_cache_v的后三维的起点。先walk过cache_v，然后walk到当前的v_head；
  T *v_cache = &params.cache_kv[params.cache_batch_size * kv_num_head * params.max_seq_length * Dh +
                                kv_bhi * params.max_seq_length * Dh];
// V_vec就是32B
  using V_vec = typename kai_V_vec_<T, V_ELTS_IN_32B>::Type;
  // 定义了宏 就是Qk_vec的每个元素都转成float
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
  using Out_vec = typename kai_Out_vec_fp32_<Qk_vec>::Type;
#else
  using Out_vec = Qk_vec;
#endif
  // 最后一个block读取当前v：当前v也要进行dot计算，但是memory bound的情况下，主要解决访存延迟。
  // 和读当前qk的思路相同
  if(tid < QK_VECS_PER_WARP){
    if ((Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) && is_last_block) {
      Qk_vec v;
      zero(v);
      Qk_vec v_bias;
      zero(v_bias);
      // Qut_vec out;
      // zero(out);

      /// TODO: beam_offset
      // kai：这个load的offset是：左移到当前的bi 再右移到v 再右移到当前的vi
      load_func.template load<Qk_vec>(v, qk_offset - hi*Dh + (params.num_head + kv_num_head) * Dh + hi / num_head_per_group * Dh);
    
      /// TODO: 加bias  
      if(params.add_qkv_bias){
        // int qkv_bias_offset = ;
        // v_bias = *reinterpret_cast<const Qk_vec*>(&params.qkv_bias[qkv_bias_offset]);
        // v = add(v, v_bias);
      }

      // 写入cache_v
      /// TODO: 零填充
      // cacheV_y 表示当前v应该写在哪行
      int cacheV_y = (end_seq - start_seq) / V_VECS_IN_32B;
      // cacheV_x 表示每个线程在后两维的offset; 
      int cacheV_x = tid * V_ELTS_IN_32B + ((end_seq - start_seq) % V_VECS_IN_32B) * QK_VEC_SIZE;
      int offset = cacheV_y * Dh * V_VECS_IN_32B + cacheV_x;
      *reinterpret_cast<Qk_vec *>(&v_cache[offset]) = v;
      
      //// 当前logits dot v 统一交给cache那里做，大部分情况下可能都是高效的？
/*
      /// TODO: *=logits[-1]
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
      /// TODO: 这里可能不对
      float logit = logtis_smem[act_time_step - start_seq];
      out = fma(logit, cast_to_float(v), out);
#else
      DataType_ logit = static_cast<DataType_>(logtis_smem[act_time_step - start_seq]);
      out = fma(logit, v, out);
#endif
      /// TODO: store result to split_out 这里不对
      int split_out_x = (bhsi + split_index) * Dh + outer_iter * THREADS_PER_BLOCK / THREADS_PER_VALUE * QK_VEC_SIZE 
                           + tid / THREADS_PER_VALUE * QK_VEC_SIZE;
      #pragma unroll
      for(int split_out_iter = 0; split_out_iter < QK_VEC_SIZE; split_out_iter++){
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
        params.split_out[split_out_x] = ((float*)(&out))[split_out_iter];
#else
        params.split_out[split_out_x] = static_cast<float>((reinterpret_cast<const DataType_*>(&out))[split_out_iter]);
#endif
      }
*/
    }
  }
  // 写和读相同cachev元素的线程不一样，需要同步一下。
  __syncthreads();

  //// 因为当前的logits dot v交给这里做，所以要end_seq + 1
  // 读logits到寄存器，cachev的外层循环可以重复使用；这里暂时没有考虑寄存器量够不够的问题(seq/THREADS_PER_VALUE)，TODO！！！
  // logits_smem循环 每个线程的开始元素的位置
  int logits_loop_begin = (tid % THREADS_PER_VALUE) * V_VECS_IN_32B;
  int logits_loop_end = end_seq - start_seq;
  // 每iter读取8 * V_VECS_IN_32B个元素
  int logits_per_iter = THREADS_PER_VALUE * V_VECS_IN_32B;
  constexpr int LOG_VEC_SIZE = steps_per_block / THREADS_PER_VALUE;
  using Logits_vec = typename kai_Logits_vec<V_VECS_IN_32B>::Type;
  Logits_vec log_vec[LOG_VEC_SIZE];
  // read logits：一个warp的8个线程 一个线程读V_VECS_IN_32B个logits 一行的4个线程通过共享内存广播共享相同的读取。float* logits_smem;
  /// TODO: 检查问题
  int log_vec_idx;
  for(logits_iter = logits_loop_begin, log_vec_idx = 0; logits_iter < logits_loop_end; logits_iter += logits_per_iter, log_vec_idx++){
    log_vec[log_vec_idx] = *(reinterpret_cast<Logits_vec>(&logits_smem[logits_iter]));
  }

  
  // 所有block读cache_v，并执行 logits dot cache_v 
  // 外层循环 有THREADS_PER_BLOCK/THREADS_PER_VALUE个线程，每个线程每iter处理QK_VEC_SIZE*V_VECS_IN_32B个元素，一行有Dh * V_VECS_IN_32B个元素
  // 一个迭代处理的元素个数
  constexpr int dh_per_iter = THREADS_PER_BLOCK / THREADS_PER_VALUE * V_ELTS_IN_32B;
  constexpr int num_outer_loop = Dh_MAX  * V_VECS_IN_32B / dh_per_iter;
  // 内层seq循环
  // 对于最后一个block来说，内存末尾元素的位置可能有些复杂。定位seq的起点和终点。
  int start_seq_forV = start_seq / V_ELTS_IN_32B;
  int inner_loop_begin = start_seq_forV + (tid % THREADS_PER_VALUE);
  int inner_loop_end = start_seq_forV + div_up(end_seq - start_seq, V_VECS_IN_32B);
  int inner_per_iter = THREADS_PER_VALUE;

  // 外层循环Dh 要考虑超过Dh的部分
  for(int outer_iter = 0; outer_iter < num_outer_loop; outer_iter++){
    int cacheV_x = outer_iter * dh_per_iter + (tid / THREADS_PER_VALUE) * V_ELTS_IN_32B;
    // 对于Dh=32或更小的情况，T=bf16, tid > 128则 tid/8*16 > 32*4。则半数warp可以提前出循环，不用冗余计算。
    if((Dh == Dh_MAX || cacheV_x < Dh / QK_VEC_SIZE * V_ELTS_IN_32B)){
      V_vec v;
      /// TODO: 思考一下这个out和最前面那个out的关系
      Out_vec out;
      // 内层循环seq/v_vecs
      for(int inner_iter = inner_loop_begin; inner_iter < inner_loop_end; inner_iter += inner_per_iter){
        int cacheV_y = inner_iter;
        int offset = cacheV_y * Dh_MAX * V_VECS_IN_32B + cacheV_x;
        // 如果cacheV_y没有超过end_seq行的范围，则成功读取，否则为零。
        v = (inner_iter < inner_loop_end) ? *reinterpret_cast<const V_vec *>(&v_cache[offset]) : v_vec_zero;
        // logits dot cache_v
        // V_vec是 V_VECS_IN_32B * QK_VEC_SIZE个元素组成的，V_VECS_IN_32B维会累加，Qk_vec会存起来
        // Out_vec out的元素个数应是QK_VEC_SIZE 数据类型为float/T
        #pragma unroll
        for(int logit_vec_idx = 0; logit_vec_idx < V_VECS_IN_32B; logit_vec_idx++){
          Qk_vec v_for_fma = (reinterpret_cast<const Qk_vec*>(&v))[logit_vec_idx];
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
          /// TODO: 这里不对
          float logit = logits_reg[logit_vec_idx];
          // 这里要做 float * cast_to_float(Qk_vec) 累加
          // Qk_vec可能是float/float2/float4/uint32_t/uint2/uint4/__nv_bfloat162/bf16_4_t/bf16_8_t
          // cast_to_float可能是float/float2/float4/Float4_/Float8_
          out = fma(logit, cast_to_float(v_for_fma), out);
#else
          DataType_ logit = static_cast<DataType_>(logits_reg[logit_vec_idx]);
          out = fma(logit, v_for_fma, out);
#endif
        } 
      }
      // store result to split_out; 
      // batch_head_split_seq_offset + outer_iter + blockLevel_offset
      int split_out_x = (bhsi + split_index) * Dh + outer_iter * THREADS_PER_BLOCK / THREADS_PER_VALUE * QK_VEC_SIZE 
                           + tid / THREADS_PER_VALUE * QK_VEC_SIZE;
      #pragma unroll
      for(int split_out_iter = 0; split_out_iter < QK_VEC_SIZE; split_out_iter++){
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
        params.split_out[split_out_x] += ((float*)(&out))[split_out_iter];
#else
        params.split_out[split_out_x] += static_cast<float>((reinterpret_cast<const DataType_*>(&out))[split_out_iter]);
#endif
      }
    }
  }



// kai mod
  constexpr int THREADS_PER_VALUE = 8;


//// for mmha_util.cu.h
/// kai mod
// for 32B + fp32 I need; 虽然就是Float8_，但是为了切割清晰，还是自己定义了一个
struct kai_Float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};
// for 32B + fp16 I need
struct kai_uint8 {
  uint2 x;
  uint2 y;
  uint2 z;
  uint2 w;
}
// for 32B + bf16 I need
#ifdef ENABLE_BF16
struct kai_bf16_16_t{
  bf16_4_t x;
  bf16_4_t y;
  bf16_4_t z;
  bf16_4_t w;
}
#endif

template <typename T, int V_VEC_SIZE>
struct kai_V_vec_ {};
template <>
struct kai_V_vec_<float, 1> {
  using Type = float;
};
template <>
struct kai_V_vec_<float, 2> {
  using Type = float2;
};
template <>
struct kai_V_vec_<float, 4> {
  using Type = float4;
};
template <>
struct kai_V_vec_<float, 8>{
  using Type = kai_Float8_;
}
template <>
struct kai_V_vec_<float16, 2> {
  using Type = uint32_t;
};
template <>
struct kai_V_vec_<float16, 4> {
  using Type = uint2;
};
template <>
struct kai_V_vec_<float16, 8> {
  using Type = uint4;
};
template <>
struct kai_V_vec_<float16, 16> {
  using Type = kai_uint8;
};
#ifdef ENABLE_BF16
template <>
struct kai_V_vec_<bfloat16, 2> {
  using Type = __nv_bfloat162;
};
template <>
struct kai_V_vec_<bfloat16, 4> {
  using Type = bf16_4_t;
};
template <>
struct kai_V_vec_<bfloat16, 8> {
  using Type = bf16_8_t;
};
template <>
struct kai_V_vec_<bfloat16, 16> {
  using Type = kai_bf16_16_t;
};
#endif  // ENABLE_BF16


template <int V_VECS_IN_32B>
struct kai_Logits_vec {};
template <> struct kai_Logits_vec<float, 1> {  using Type = float; };
template <> struct kai_Logits_vec<float, 2> {  using Type = float2; };
template <> struct kai_Logits_vec<float, 4> {  using Type = float4; };
template <> struct kai_Logits_vec<float, 8> {  using Type = Float8_; };

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
template <typename T>
struct kai_Out_vec_fp32_ {};
template <> struct kai_Out_vec_fp32_<float>  { using Type = float;  };
template <> struct kai_Out_vec_fp32_<float2> { using Type = float2; };
template <> struct kai_Out_vec_fp32_<float4> { using Type = float4; };
template <> struct kai_Out_vec_fp32_<uint32_t> { using Type = float2; };
template <> struct kai_Out_vec_fp32_<uint2   > { using Type = Float4_; };
template <> struct kai_Out_vec_fp32_<uint4> {  using Type = Float8_; };
#ifdef ENABLE_BF16
template <> struct kai_Out_vec_fp32_<__nv_bfloat162> {  using Type = float2; };
template <> struct kai_Out_vec_fp32_<bf16_4_t> {  using Type = Float4_; };
template <> struct kai_Out_vec_fp32_<bf16_8_t> {  using Type = Float8_; };
#endif  // ENABLE_BF16
#endif
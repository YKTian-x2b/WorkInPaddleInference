
@triton.jit
def fused_moe_kernel(a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    topK_M,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(topK_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(GROUP_SIZE_M, num_pid_m-first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # sorted_token_ids_ptr确保了padding_token是可以整除BLOCK_SIZE_M的，这样的话，超长退出就是合理的。
    if pid_m * BLOCK_SIZE_M >= tl.load(num_tokens_post_padded_ptr):
        return
    # 如果按专家来读token
    offset_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_token = tl.load(sorted_token_ids_ptr + offset_am)
    token_mask = offset_token < num_valid_tokens
    
    offset_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_be = tl.load(expert_ids_ptr + pid_m)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offset_token[:, None] * stride_am + offset_k[None, :] * stride_ak)
    b_ptrs = b_ptr + offset_be * stride_be + (offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = token_mask[:,None] + offset_k[None, :] < K-k*BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=a_mask, other=0.)
        b = tl.load(b_ptrs, mask=offset_k[:, None] < K-k*BLOCK_SIZE_K, other=0.)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    moe_weight = tl.load(topk_weights_ptr)
    accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)
    # store C


def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool, top_k: int,
                            config: Dict[str, Any]) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
        'BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']), )

    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16,
        **config,
    )
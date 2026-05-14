import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _decode_attention_kernel(
        q_ptr,
        past_k_ptr,
        past_v_ptr,
        new_k_ptr,
        new_v_ptr,
        out_ptr,
        past_len,
        head_dim: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_t: tl.constexpr,
        q_stride_d: tl.constexpr,
        pk_stride_b: tl.constexpr,
        pk_stride_h: tl.constexpr,
        pk_stride_t: tl.constexpr,
        pk_stride_d: tl.constexpr,
        pv_stride_b: tl.constexpr,
        pv_stride_h: tl.constexpr,
        pv_stride_t: tl.constexpr,
        pv_stride_d: tl.constexpr,
        nk_stride_b: tl.constexpr,
        nk_stride_h: tl.constexpr,
        nk_stride_t: tl.constexpr,
        nk_stride_d: tl.constexpr,
        nv_stride_b: tl.constexpr,
        nv_stride_h: tl.constexpr,
        nv_stride_t: tl.constexpr,
        nv_stride_d: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_t: tl.constexpr,
        out_stride_d: tl.constexpr,
        n_head: tl.constexpr,
        scale: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        b = program_id // n_head
        h = program_id - b * n_head

        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim

        q = tl.load(
            q_ptr
            + b * q_stride_b
            + h * q_stride_h
            + offs_d * q_stride_d,
            mask=d_mask,
            other=0.0,
        )

        m = tl.full((), -float("inf"), tl.float32)
        l = tl.full((), 0.0, tl.float32)
        acc = tl.full((block_d,), 0.0, tl.float32)

        start = 0
        while start < past_len:
            offs_n = start + tl.arange(0, block_n)
            n_mask = offs_n < past_len
            k = tl.load(
                past_k_ptr
                + b * pk_stride_b
                + h * pk_stride_h
                + offs_n[:, None] * pk_stride_t
                + offs_d[None, :] * pk_stride_d,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            v = tl.load(
                past_v_ptr
                + b * pv_stride_b
                + h * pv_stride_h
                + offs_n[:, None] * pv_stride_t
                + offs_d[None, :] * pv_stride_d,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            scores = tl.sum(k * q[None, :], axis=1) * scale
            scores = tl.where(n_mask, scores, -float("inf"))

            block_m = tl.max(scores, axis=0)
            new_m = tl.maximum(m, block_m)
            alpha = tl.exp(m - new_m)
            probs = tl.exp(scores - new_m)
            acc = acc * alpha + tl.sum(probs[:, None] * v, axis=0)
            l = l * alpha + tl.sum(probs, axis=0)
            m = new_m
            start += block_n

        new_k = tl.load(
            new_k_ptr
            + b * nk_stride_b
            + h * nk_stride_h
            + offs_d * nk_stride_d,
            mask=d_mask,
            other=0.0,
        )
        new_v = tl.load(
            new_v_ptr
            + b * nv_stride_b
            + h * nv_stride_h
            + offs_d * nv_stride_d,
            mask=d_mask,
            other=0.0,
        )
        score = tl.sum(new_k * q, axis=0) * scale
        new_m = tl.maximum(m, score)
        alpha = tl.exp(m - new_m)
        prob = tl.exp(score - new_m)
        acc = acc * alpha + prob * new_v
        l = l * alpha + prob

        out = acc / l
        tl.store(
            out_ptr
            + b * out_stride_b
            + h * out_stride_h
            + offs_d * out_stride_d,
            out,
            mask=d_mask,
        )


def decode_attention_direct(
    q: torch.Tensor,
    past_k: torch.Tensor,
    past_v: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
) -> torch.Tensor:
    if (
        triton is None
        or not q.is_cuda
        or q.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        k = torch.cat([past_k, new_k], dim=2)
        v = torch.cat([past_v, new_v], dim=2)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

    batch_size, n_head, query_len, head_dim = q.shape
    if query_len != 1:
        raise ValueError("decode_attention_direct only supports query_len == 1")

    past_len = past_k.size(2)
    if head_dim > 128:
        k = torch.cat([past_k, new_k], dim=2)
        v = torch.cat([past_v, new_v], dim=2)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

    out = torch.empty_like(q)
    block_d = triton.next_power_of_2(head_dim)
    block_n = 64
    grid = (batch_size * n_head,)
    _decode_attention_kernel[grid](
        q,
        past_k,
        past_v,
        new_k,
        new_v,
        out,
        past_len,
        head_dim,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        past_k.stride(0),
        past_k.stride(1),
        past_k.stride(2),
        past_k.stride(3),
        past_v.stride(0),
        past_v.stride(1),
        past_v.stride(2),
        past_v.stride(3),
        new_k.stride(0),
        new_k.stride(1),
        new_k.stride(2),
        new_k.stride(3),
        new_v.stride(0),
        new_v.stride(1),
        new_v.stride(2),
        new_v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        n_head,
        1.0 / math.sqrt(head_dim),
        block_n,
        block_d,
        num_warps=4,
    )
    return out

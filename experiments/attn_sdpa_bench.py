import torch
import torch.nn.functional as F
import argparse

def old_attention(q, k, v, causal=True):
    D = q.size(-1)
    att = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)

    if causal:
        Tq, Tk = q.size(-2), k.size(-2)
        mask = torch.tril(torch.ones(Tq, Tk, device=q.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))

    att = torch.softmax(att, dim=-1)
    return torch.matmul(att, v)


def sdpa_attention(q, k, v):
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True
    )


def benchmark(fn, q, k, v, iters, warmup, label):
    # warmup
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out = fn(q, k, v)
    end.record()

    torch.cuda.synchronize()
    time_ms = start.elapsed_time(end) / iters
    print(f"{label}: {time_ms:.4f} ms")


def run_case(mode, B, H, Tq, Tk, D, dtype, device, iters, warmup):
    print(f"\n=== Mode={mode} | B={B}, H={H}, Tq={Tq}, Tk={Tk}, D={D} ===")

    q = torch.randn(B, H, Tq, D, device=device, dtype=dtype)
    k = torch.randn(B, H, Tk, D, device=device, dtype=dtype)
    v = torch.randn(B, H, Tk, D, device=device, dtype=dtype)

    if mode == "old":
        benchmark(old_attention, q, k, v, iters, warmup, "OLD attention")

    elif mode == "sdpa":
        # force fused kernels where possible
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True
        ):
            benchmark(sdpa_attention, q, k, v, iters, warmup, "SDPA")

    else:
        raise ValueError("mode must be 'old' or 'sdpa'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["old", "sdpa"])
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)

    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = "cuda"

    assert torch.cuda.is_available()

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 🔹 Prefill (GEMM-like)
    run_case(
        args.mode,
        args.B, args.H,
        Tq=512, Tk=512,
        D=args.D,
        dtype=dtype,
        device=device,
        iters=args.iters,
        warmup=args.warmup
    )

    # 🔹 Decode (GEMV-like)
    run_case(
        args.mode,
        args.B, args.H,
        Tq=1, Tk=2048,
        D=args.D,
        dtype=dtype,
        device=device,
        iters=args.iters,
        warmup=args.warmup
    )
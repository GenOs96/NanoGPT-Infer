import torch
import time
import argparse

def bench(B, H, S, D, dtype, device, iters, warmup):
    torch.manual_seed(0)

    q = torch.randn(B, H, 1, D, device=device, dtype=dtype)

    # Old layout: (B, H, S, D)
    k_old = torch.randn(B, H, S, D, device=device, dtype=dtype)

    # New layout: (B, H, D, S)
    #k_new = k_old.transpose(-2, -1).contiguous()

    # Warmup
    for _ in range(warmup):
        _ = torch.matmul(q, k_old.transpose(-2, -1))
        #_ = torch.matmul(q, k_new)

    torch.cuda.synchronize()

    # Timing helper
    def run(fn, label):
        start = time.time()
        for _ in range(iters):
            out = fn()
        torch.cuda.synchronize()
        end = time.time()
        print(f"{label}: {(end - start)*1000/iters:.4f} ms")

    print("\n=== Benchmark ===")
    run(lambda: torch.matmul(q, k_old.transpose(-2, -1)), "OLD layout (B,H,S,D)")
    #run(lambda: torch.matmul(q, k_new), "NEW layout (B,H,D,S)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--S", type=int, default=2048)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="fp16")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    assert torch.cuda.is_available()
    device = "cuda"

    print(f"Running on {torch.cuda.get_device_name(0)}")
    print(f"B={args.B}, H={args.H}, S={args.S}, D={args.D}, dtype={dtype}")

    bench(args.B, args.H, args.S, args.D, dtype, device, args.iters, args.warmup)
import torch
import torch.nn.functional as F
import argparse

# -----------------------------
# KV CACHE IMPLEMENTATIONS
# -----------------------------

class KVCacheConcat:
    def __init__(self):
        self.k = None
        self.v = None

    def update(self, k_new, v_new):
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v


class KVCachePrealloc:
    def __init__(self, B, H, max_seq, D, device, dtype):
        self.k = torch.empty(B, H, max_seq, D, device=device, dtype=dtype)
        self.v = torch.empty(B, H, max_seq, D, device=device, dtype=dtype)
        self.pos = 0

    def update(self, k_new, v_new):
        T = k_new.size(2)
        self.k[:, :, self.pos:self.pos+T, :] = k_new
        self.v[:, :, self.pos:self.pos+T, :] = v_new
        self.pos += T
        return self.k[:, :, :self.pos], self.v[:, :, :self.pos]


class KVCacheSDPAOptimized:
    def __init__(self, B, H, max_seq, D, device, dtype):
        self.k = torch.empty(B, H, max_seq, D, device=device, dtype=dtype)
        self.v = torch.empty(B, H, max_seq, D, device=device, dtype=dtype)
        self.pos = 0

    def update(self, k_new, v_new):
        T = k_new.size(2)

        # avoids extra view creation
        self.k.narrow(2, self.pos, T).copy_(k_new)
        self.v.narrow(2, self.pos, T).copy_(v_new)

        self.pos += T

        return self.k[:, :, :self.pos], self.v[:, :, :self.pos]


# -----------------------------
# ATTENTION
# -----------------------------

def sdpa(q, k, v):
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True
    )


# -----------------------------
# BENCHMARK
# -----------------------------

def run_decode(cache, B, H, D, steps, device, dtype):
    total_time = 0.0

    for _ in range(steps):
        q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        k_new = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        v_new = torch.randn(B, H, 1, D, device=device, dtype=dtype)

        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)

        start.record()

        k, v = cache.update(k_new, v_new)
        _ = sdpa(q, k, v)

        end.record()
        torch.cuda.synchronize()

        total_time += start.elapsed_time(end)

    return total_time / steps


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["concat", "prealloc", "optimized"])
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="fp16")

    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = "cuda"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mode: {args.mode}")

    if args.mode == "concat":
        cache = KVCacheConcat()

    elif args.mode == "prealloc":
        cache = KVCachePrealloc(args.B, args.H, args.steps, args.D, device, dtype)

    elif args.mode == "optimized":
        cache = KVCacheSDPAOptimized(args.B, args.H, args.steps, args.D, device, dtype)

    else:
        raise ValueError("Invalid mode")

    time_ms = run_decode(cache, args.B, args.H, args.D, args.steps, device, dtype)

    print(f"\nAverage step time: {time_ms:.4f} ms")
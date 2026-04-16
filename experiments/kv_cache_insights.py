import math
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from inference.generate import generate
from inference.kv_cache import KVCache
from inference.sampler import sample_logits
from model.gpt import GPT, GPTConfig, load_hf_weights


MODEL_NAME = "gpt2"
DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIXED_TOTAL_CONTEXT_LEN = 1024
WARMUP_RUNS = 2
MEASURE_RUNS = 5
SEED = 1234


def get_artifacts_dir() -> str:
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    return artifacts_dir


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def make_input_ids(vocab_size, context_len, batch_size, device, seed=None):
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return torch.randint(
            0,
            vocab_size,
            (batch_size, context_len),
            device=device,
            generator=generator,
        )

    return torch.randint(0, vocab_size, (batch_size, context_len), device=device)


def build_model():
    set_seed(SEED)

    hf_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    hf_model.eval()

    cfg = GPTConfig(
        vocab_size=hf_model.config.vocab_size,
        block_size=hf_model.config.n_positions,
        n_layer=hf_model.config.n_layer,
        n_head=hf_model.config.n_head,
        n_embd=hf_model.config.n_embd,
    )

    model = GPT(cfg)
    load_hf_weights(model, hf_model)
    model = model.to(DEVICE, dtype=DTYPE)
    model.eval()
    return model, hf_model.config.vocab_size


def build_kv_cache(model, batch_size, max_context_len):
    return KVCache(
        n_layer=model.config.n_layer,
        batch_size=batch_size,
        n_head=model.config.n_head,
        max_seq_len=max_context_len,
        head_dim=model.config.n_embd // model.config.n_head,
        device=DEVICE,
    )


def benchmark_end_to_end(
    model,
    idx,
    use_kv_cache,
    max_context_len,
    warmup_runs=2,
    measure_runs=5,
    base_seed=2026,
):
    generated_tokens = max_context_len - idx.shape[1]

    for i in range(warmup_runs):
        set_seed(base_seed + i)
        _ = generate(
            model,
            idx.clone(),
            use_kv_cache=use_kv_cache,
            max_context_len=max_context_len,
        )

    timings = []
    for i in range(measure_runs):
        set_seed(base_seed + 1000 + i)
        x = idx.clone()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = generate(
            model,
            x,
            use_kv_cache=use_kv_cache,
            max_context_len=max_context_len,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)

    throughput = [
        (idx.shape[0] * generated_tokens) / t for t in timings
    ] if generated_tokens > 0 else [0.0 for _ in timings]
    return timings, throughput, generated_tokens


def benchmark_prefill_only(
    model,
    idx,
    use_kv_cache,
    max_context_len,
    warmup_runs=2,
    measure_runs=5,
    base_seed=3030,
):
    prompt_tokens = idx.shape[0] * idx.shape[1]

    for i in range(warmup_runs):
        set_seed(base_seed + i)
        kv_cache = build_kv_cache(model, idx.shape[0], max_context_len) if use_kv_cache else None
        _ = model(idx.clone(), kv_cache=kv_cache)

    timings = []
    for i in range(measure_runs):
        set_seed(base_seed + 1000 + i)
        x = idx.clone()
        kv_cache = build_kv_cache(model, x.shape[0], max_context_len) if use_kv_cache else None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x, kv_cache=kv_cache)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)

    throughput = [prompt_tokens / t for t in timings] if prompt_tokens > 0 else [0.0 for _ in timings]
    return timings, throughput


def benchmark_decode_only(
    model,
    idx,
    use_kv_cache,
    max_context_len,
    warmup_runs=2,
    measure_runs=5,
    base_seed=4040,
):
    generated_tokens = max_context_len - idx.shape[1]
    if generated_tokens <= 0:
        return [0.0 for _ in range(measure_runs)], [0.0 for _ in range(measure_runs)], 0

    for i in range(warmup_runs):
        set_seed(base_seed + i)
        x = idx.clone()
        if use_kv_cache:
            kv_cache = build_kv_cache(model, x.shape[0], max_context_len)
            logits = model(x, kv_cache=kv_cache)
            for _ in range(generated_tokens):
                next_token = sample_logits(logits)
                logits = model(next_token, kv_cache=kv_cache)
        else:
            for _ in range(generated_tokens):
                logits = model(x)
                next_token = sample_logits(logits)
                x = torch.cat([x, next_token], dim=1)

    timings = []
    for i in range(measure_runs):
        set_seed(base_seed + 1000 + i)
        x = idx.clone()
        if use_kv_cache:
            kv_cache = build_kv_cache(model, x.shape[0], max_context_len)
            logits = model(x, kv_cache=kv_cache)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(generated_tokens):
                next_token = sample_logits(logits)
                logits = model(next_token, kv_cache=kv_cache)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(generated_tokens):
                logits = model(x)
                next_token = sample_logits(logits)
                x = torch.cat([x, next_token], dim=1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)

    throughput = [(idx.shape[0] * generated_tokens) / t for t in timings]
    return timings, throughput, generated_tokens


def kv_cache_memory_mb(model, batch_size, seq_len, dtype=None):
    if dtype is None:
        dtype = next(model.parameters()).dtype
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    head_dim = model.config.n_embd // model.config.n_head
    values = 2 * model.config.n_layer * batch_size * seq_len * model.config.n_head * head_dim
    return values * bytes_per_elem / (1024 ** 2)


def prompt_length_study(model, vocab_size):
    rows = []
    prompt_lengths = [64, 128, 256, 512, 768, 896, 1024]

    for context_len in prompt_lengths:
        idx = make_input_ids(vocab_size, context_len, 1, DEVICE, seed=SEED + context_len)
        for mode_name, use_kv in [("no_cache", False), ("kv_cache_v1", True)]:
            timings, _, generated_tokens = benchmark_end_to_end(
                model,
                idx,
                use_kv_cache=use_kv,
                max_context_len=FIXED_TOTAL_CONTEXT_LEN,
                warmup_runs=WARMUP_RUNS,
                measure_runs=MEASURE_RUNS,
                base_seed=SEED + 500,
            )
            rows.append(
                {
                    "mode": mode_name,
                    "context_len": context_len,
                    "new_tokens": generated_tokens,
                    "e2e_latency_s_mean": summarize(timings)["mean"],
                }
            )

    df = pd.DataFrame(rows)
    print("\nEnd-to-end latency vs prompt length")
    print(df.sort_values(["context_len", "mode"]).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in ["no_cache", "kv_cache_v1"]:
        mode_df = df[df["mode"] == mode].sort_values("context_len")
        ax.plot(mode_df["context_len"], mode_df["e2e_latency_s_mean"], marker="o", linewidth=2, label=mode)

    ax.set_title("End-to-End Latency vs Prompt Length (batch=1)")
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("Latency (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    png_path = os.path.join(get_artifacts_dir(), "kv_cache_prompt_length_study.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.show()
    plt.close(fig)
    return df


def batch_size_study(model, vocab_size):
    rows = []
    prompt_len_for_batch = 128
    batch_sizes = [1, 2, 4, 8, 16]

    for batch_size in batch_sizes:
        idx = make_input_ids(
            vocab_size,
            prompt_len_for_batch,
            batch_size,
            DEVICE,
            seed=SEED + batch_size + 2000,
        )
        for mode_name, use_kv in [("no_cache", False), ("kv_cache_v1", True)]:
            timings, throughput, generated_tokens = benchmark_end_to_end(
                model,
                idx,
                use_kv_cache=use_kv,
                max_context_len=FIXED_TOTAL_CONTEXT_LEN,
                warmup_runs=WARMUP_RUNS,
                measure_runs=MEASURE_RUNS,
                base_seed=SEED + 2500,
            )
            rows.append(
                {
                    "mode": mode_name,
                    "batch_size": batch_size,
                    "context_len": prompt_len_for_batch,
                    "new_tokens": generated_tokens,
                    "throughput_tok_s_mean": summarize(throughput)["mean"],
                    "e2e_latency_s_mean": summarize(timings)["mean"],
                }
            )

    df = pd.DataFrame(rows)
    print("\nThroughput vs batch size")
    print(df.sort_values(["batch_size", "mode"]).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in ["no_cache", "kv_cache_v1"]:
        mode_df = df[df["mode"] == mode].sort_values("batch_size")
        ax.plot(mode_df["batch_size"], mode_df["throughput_tok_s_mean"], marker="o", linewidth=2, label=mode)

    ax.set_title("Throughput vs Batch Size (prompt=128)")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_xticks(batch_sizes)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    png_path = os.path.join(get_artifacts_dir(), "kv_cache_batch_size_study.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.show()
    plt.close(fig)
    return df


def phase_study(model, vocab_size):
    rows = []
    phase_prompt_len = 512
    phase_idx = make_input_ids(vocab_size, phase_prompt_len, 1, DEVICE, seed=SEED + 7000)

    for mode_name, use_kv in [("no_cache", False), ("kv_cache_v1", True)]:
        prefill_timings, prefill_throughput = benchmark_prefill_only(
            model,
            phase_idx,
            use_kv_cache=use_kv,
            max_context_len=FIXED_TOTAL_CONTEXT_LEN,
            warmup_runs=WARMUP_RUNS,
            measure_runs=MEASURE_RUNS,
            base_seed=SEED + 7100,
        )
        decode_timings, decode_throughput, generated_tokens = benchmark_decode_only(
            model,
            phase_idx,
            use_kv_cache=use_kv,
            max_context_len=FIXED_TOTAL_CONTEXT_LEN,
            warmup_runs=WARMUP_RUNS,
            measure_runs=MEASURE_RUNS,
            base_seed=SEED + 7200,
        )

        rows.append(
            {
                "mode": mode_name,
                "phase": "prefill",
                "token_count": phase_prompt_len,
                "latency_s_mean": summarize(prefill_timings)["mean"],
                "throughput_tok_s_mean": summarize(prefill_throughput)["mean"],
            }
        )
        rows.append(
            {
                "mode": mode_name,
                "phase": "decode",
                "token_count": generated_tokens,
                "latency_s_mean": summarize(decode_timings)["mean"],
                "throughput_tok_s_mean": summarize(decode_throughput)["mean"],
            }
        )

    df = pd.DataFrame(rows)
    print("\nPrefill vs decode")
    print(df.sort_values(["phase", "mode"]).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"no_cache": "#d95f02", "kv_cache_v1": "#1b9e77"}
    phases = ["prefill", "decode"]
    x = np.arange(len(phases))
    width = 0.35

    for offset, mode in [(-width / 2, "no_cache"), (width / 2, "kv_cache_v1")]:
        mode_df = df[df["mode"] == mode].set_index("phase").loc[phases]
        axes[0].bar(x + offset, mode_df["latency_s_mean"], width=width, color=colors[mode], label=mode)
        axes[1].bar(x + offset, mode_df["throughput_tok_s_mean"], width=width, color=colors[mode], label=mode)

    axes[0].set_title(f"Prefill vs Decode Latency (prompt={phase_prompt_len}, batch=1)")
    axes[0].set_ylabel("Latency (s)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(phases)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].set_title(f"Prefill vs Decode Throughput (prompt={phase_prompt_len}, batch=1)")
    axes[1].set_ylabel("Throughput (tokens/sec)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(phases)
    axes[1].grid(True, axis="y", alpha=0.25)

    for ax in axes:
        ax.legend()

    plt.tight_layout()
    png_path = os.path.join(get_artifacts_dir(), "kv_cache_phase_study.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.show()
    plt.close(fig)
    return df


def memory_study(model):
    seq_lengths = [64, 128, 256, 512, 768, 896, 1024]
    warp_size = 32
    dtype = next(model.parameters()).dtype
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    head_dim = model.config.n_embd // model.config.n_head
    warps_per_head_vector = math.ceil(head_dim / warp_size)
    bytes_per_head_vector = head_dim * bytes_per_elem

    rows = []
    for seq_len in seq_lengths:
        rows.append(
            {
                "seq_len": seq_len,
                "active_kv_mb": kv_cache_memory_mb(model, batch_size=1, seq_len=seq_len, dtype=dtype),
                "allocated_kv_mb": kv_cache_memory_mb(model, batch_size=1, seq_len=FIXED_TOTAL_CONTEXT_LEN, dtype=dtype),
                "decode_warp_groups_per_layer": seq_len * model.config.n_head * warps_per_head_vector,
                "decode_bytes_read_per_layer_mb": (
                    2 * seq_len * model.config.n_head * head_dim * bytes_per_elem
                ) / (1024 ** 2),
            }
        )

    df = pd.DataFrame(rows)
    print("\nKV-cache memory growth and decode access model")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(df["seq_len"], df["active_kv_mb"], marker="o", linewidth=2, label="Active KV memory")
    axes[0].plot(
        df["seq_len"],
        df["allocated_kv_mb"],
        linestyle="--",
        linewidth=2,
        label="Allocated KV memory @ 1024",
    )
    axes[0].set_title("KV-Cache Memory Growth")
    axes[0].set_xlabel("Sequence length")
    axes[0].set_ylabel("Memory (MB)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        df["seq_len"],
        df["decode_warp_groups_per_layer"],
        marker="o",
        linewidth=2,
        label="Warp groups per decode step / layer",
    )
    axes[1].plot(
        df["seq_len"],
        df["decode_bytes_read_per_layer_mb"],
        marker="s",
        linewidth=2,
        label="Bytes read per decode step / layer (MB)",
    )
    axes[1].set_title("Decode Access Growth with Context")
    axes[1].set_xlabel("Sequence length")
    axes[1].set_ylabel("Analytical cost")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    png_path = os.path.join(get_artifacts_dir(), "kv_cache_memory_study.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.show()
    plt.close(fig)

    print(f"dtype={dtype}, bytes_per_elem={bytes_per_elem}")
    print(f"head_dim={head_dim}, n_head={model.config.n_head}, n_layer={model.config.n_layer}")
    print(
        f"bytes_per_head_vector={bytes_per_head_vector}, "
        f"warps_per_head_vector={warps_per_head_vector}"
    )
    return df


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(torch.cuda.get_device_name(0))

    model, vocab_size = build_model()
    assert FIXED_TOTAL_CONTEXT_LEN <= model.config.block_size

    prompt_df = prompt_length_study(model, vocab_size)
    batch_df = batch_size_study(model, vocab_size)
    phase_df = phase_study(model, vocab_size)
    memory_df = memory_study(model)

    artifacts_dir = get_artifacts_dir()
    prompt_df.to_csv(os.path.join(artifacts_dir, "kv_cache_prompt_length_study.csv"), index=False)
    batch_df.to_csv(os.path.join(artifacts_dir, "kv_cache_batch_size_study.csv"), index=False)
    phase_df.to_csv(os.path.join(artifacts_dir, "kv_cache_phase_study.csv"), index=False)
    memory_df.to_csv(os.path.join(artifacts_dir, "kv_cache_memory_study.csv"), index=False)


if __name__ == "__main__":
    main()

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
from model.gpt import GPT, GPTConfig, load_hf_weights


MODEL_NAME = "gpt2"
DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FIXED_TOTAL_CONTEXT_LEN = 1024
CONTEXT_LENGTHS = [64, 128, 256, 512, 768, 896, 1024]
BATCH_SIZE = 1
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


def benchmark_generate(
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
    peak_mem_mb = []

    for i in range(measure_runs):
        set_seed(base_seed + 1000 + i)
        x = idx.clone()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        _ = generate(
            model,
            x,
            use_kv_cache=use_kv_cache,
            max_context_len=max_context_len,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        timings.append(t1 - t0)

        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_mem_mb.append(float(peak_mb))

    return timings, peak_mem_mb, generated_tokens


def run_correctness_gates(model, vocab_size):
    correctness_results = []
    gate_contexts = [64, 256]
    gate_batch = 1

    for context_len in gate_contexts:
        generated_tokens = FIXED_TOTAL_CONTEXT_LEN - context_len
        if generated_tokens < 0:
            correctness_results.append(
                {
                    "context_len": context_len,
                    "new_tokens": generated_tokens,
                    "batch_size": gate_batch,
                    "status": "SKIPPED_BLOCK_SIZE",
                }
            )
            continue

        idx = make_input_ids(
            vocab_size,
            context_len,
            gate_batch,
            DEVICE,
            seed=SEED + context_len + generated_tokens,
        )

        set_seed(9000)
        out_no_cache = generate(
            model,
            idx.clone(),
            use_kv_cache=False,
            max_context_len=FIXED_TOTAL_CONTEXT_LEN,
        )

        set_seed(9000)
        out_cache = generate(
            model,
            idx.clone(),
            use_kv_cache=True,
            max_context_len=FIXED_TOTAL_CONTEXT_LEN,
        )

        correctness_results.append(
            {
                "context_len": context_len,
                "new_tokens": generated_tokens,
                "batch_size": gate_batch,
                "total_context_len": FIXED_TOTAL_CONTEXT_LEN,
                "status": "PASS" if torch.equal(out_no_cache, out_cache) else "FAIL",
            }
        )

    correctness_df = pd.DataFrame(correctness_results)
    print("\nCorrectness gates")
    print(correctness_df.to_string(index=False))
    return correctness_df


def run_benchmark_sweep(model, vocab_size):
    results = []

    for context_len in CONTEXT_LENGTHS:
        generated_tokens = FIXED_TOTAL_CONTEXT_LEN - context_len

        if generated_tokens < 0:
            for mode_name in ["no_cache", "kv_cache_v1"]:
                results.append(
                    {
                        "mode": mode_name,
                        "context_len": context_len,
                        "batch_size": BATCH_SIZE,
                        "new_tokens": generated_tokens,
                        "total_context_len": FIXED_TOTAL_CONTEXT_LEN,
                        "status": "SKIPPED_BLOCK_SIZE",
                    }
                )
            continue

        idx = make_input_ids(
            vocab_size,
            context_len,
            BATCH_SIZE,
            DEVICE,
            seed=SEED + context_len + BATCH_SIZE + generated_tokens,
        )

        for mode_name, use_kv in [("no_cache", False), ("kv_cache_v1", True)]:
            timings, peak_mem, generated_tokens = benchmark_generate(
                model=model,
                idx=idx,
                use_kv_cache=use_kv,
                max_context_len=FIXED_TOTAL_CONTEXT_LEN,
                warmup_runs=WARMUP_RUNS,
                measure_runs=MEASURE_RUNS,
                base_seed=SEED,
            )

            latency_stats = summarize(timings)
            if generated_tokens > 0:
                decode_ms_per_token = [t * 1000.0 / generated_tokens for t in timings]
                throughput_toks_s = [
                    (BATCH_SIZE * generated_tokens) / t for t in timings
                ]
            else:
                decode_ms_per_token = [0.0 for _ in timings]
                throughput_toks_s = [0.0 for _ in timings]

            decode_stats = summarize(decode_ms_per_token)
            throughput_stats = summarize(throughput_toks_s)

            row = {
                "mode": mode_name,
                "context_len": context_len,
                "batch_size": BATCH_SIZE,
                "new_tokens": generated_tokens,
                "total_context_len": FIXED_TOTAL_CONTEXT_LEN,
                "status": "OK",
                "e2e_latency_s_mean": latency_stats["mean"],
                "e2e_latency_s_p50": latency_stats["p50"],
                "e2e_latency_s_p95": latency_stats["p95"],
                "decode_ms_token_mean": decode_stats["mean"],
                "decode_ms_token_p50": decode_stats["p50"],
                "decode_ms_token_p95": decode_stats["p95"],
                "throughput_tok_s_mean": throughput_stats["mean"],
                "throughput_tok_s_p50": throughput_stats["p50"],
                "throughput_tok_s_p95": throughput_stats["p95"],
            }

            if peak_mem:
                mem_stats = summarize(peak_mem)
                row["peak_mem_mb_mean"] = mem_stats["mean"]
                row["peak_mem_mb_p50"] = mem_stats["p50"]
                row["peak_mem_mb_p95"] = mem_stats["p95"]

            results.append(row)

    df = pd.DataFrame(results)
    print(f"\nCollected rows: {len(df)}")
    print(df.head(20).to_string(index=False))
    return df


def plot_sweep(df):
    ok_df = df[df["status"] == "OK"].copy()
    if ok_df.empty:
        print("No successful benchmark rows to plot.")
        return

    summary = ok_df.sort_values("context_len")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for mode in summary["mode"].unique():
        mode_df = summary[summary["mode"] == mode].sort_values("context_len")
        axes[0].plot(
            mode_df["context_len"],
            mode_df["decode_ms_token_mean"],
            marker="o",
            label=mode,
        )
        axes[1].plot(
            mode_df["context_len"],
            mode_df["throughput_tok_s_mean"],
            marker="o",
            label=mode,
        )

    axes[0].set_title("Decode Latency vs Prompt Length")
    axes[0].set_xlabel("Prompt length")
    axes[0].set_ylabel("Decode latency (ms/token)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Decode Throughput vs Prompt Length")
    axes[1].set_xlabel("Prompt length")
    axes[1].set_ylabel("Throughput (tokens/sec)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    png_path = os.path.join(get_artifacts_dir(), "kv_cache_benchmark_sweep.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.show()
    plt.close(fig)


def plot_comparison(df):
    ok_df = df[df["status"] == "OK"].copy()
    if ok_df.empty:
        print("No successful benchmark rows to plot.")
        return

    baseline = ok_df[ok_df["mode"] == "no_cache"].copy()
    cached = ok_df[ok_df["mode"] == "kv_cache_v1"].copy()

    comparison = baseline.merge(
        cached,
        on=["context_len", "batch_size", "new_tokens", "total_context_len"],
        suffixes=("_no_cache", "_kv_cache"),
    )

    comparison["latency_speedup_x"] = (
        comparison["decode_ms_token_mean_no_cache"]
        / comparison["decode_ms_token_mean_kv_cache"]
    )
    comparison["throughput_speedup_x"] = (
        comparison["throughput_tok_s_mean_kv_cache"]
        / comparison["throughput_tok_s_mean_no_cache"]
    )

    display_cols = [
        "context_len",
        "new_tokens",
        "decode_ms_token_mean_no_cache",
        "decode_ms_token_mean_kv_cache",
        "latency_speedup_x",
        "throughput_speedup_x",
    ]
    print("\nComparison summary")
    print(comparison[display_cols].sort_values("context_len").to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex="col")
    colors = {"no_cache": "#d95f02", "kv_cache_v1": "#1b9e77"}

    for mode in ["no_cache", "kv_cache_v1"]:
        mode_df = ok_df[ok_df["mode"] == mode].sort_values("context_len")
        axes[0, 0].plot(
            mode_df["context_len"],
            mode_df["decode_ms_token_mean"],
            marker="o",
            color=colors[mode],
            linewidth=2,
            label=mode,
        )
        axes[0, 1].plot(
            mode_df["context_len"],
            mode_df["throughput_tok_s_mean"],
            marker="o",
            color=colors[mode],
            linewidth=2,
            label=mode,
        )

    speed_df = comparison.sort_values("context_len")
    axes[1, 0].plot(
        speed_df["context_len"],
        speed_df["latency_speedup_x"],
        marker="o",
        color="#7570b3",
        linewidth=2,
        label="latency speedup",
    )
    axes[1, 1].plot(
        speed_df["context_len"],
        speed_df["throughput_speedup_x"],
        marker="o",
        color="#e7298a",
        linewidth=2,
        label="throughput speedup",
    )

    axes[0, 0].set_title("Decode latency by prompt length")
    axes[0, 0].set_ylabel("ms/token")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].set_title("Throughput by prompt length")
    axes[0, 1].set_ylabel("tokens/sec")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].set_title("KV-cache decode latency speedup")
    axes[1, 0].set_xlabel("Prompt length")
    axes[1, 0].set_ylabel("speedup (x)")
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].set_title("KV-cache throughput speedup")
    axes[1, 1].set_xlabel("Prompt length")
    axes[1, 1].set_ylabel("speedup (x)")
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].grid(True, alpha=0.25)

    for ax in axes.flat:
        ax.legend(fontsize=9)

    best = comparison.sort_values("throughput_speedup_x", ascending=False).iloc[0]
    fig.suptitle(
        (
            "KV-cache benchmark: "
            f"best throughput gain {best['throughput_speedup_x']:.2f}x "
            f"at prompt={int(best['context_len'])}, "
            f"generated={int(best['new_tokens'])}, "
            f"total={int(best['total_context_len'])}"
        ),
        fontsize=14,
    )
    fig.tight_layout()
    png_path = os.path.join(get_artifacts_dir(), "kv_cache_benchmark_comparison.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.show()
    plt.close(fig)


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(torch.cuda.get_device_name(0))

    model, vocab_size = build_model()
    assert FIXED_TOTAL_CONTEXT_LEN <= model.config.block_size
    print(f"Model block_size: {model.config.block_size}")
    print(f"Fixed total context length: {FIXED_TOTAL_CONTEXT_LEN}")

    correctness_df = run_correctness_gates(model, vocab_size)
    df = run_benchmark_sweep(model, vocab_size)

    ok_df = df[df["status"] == "OK"].copy()
    if not ok_df.empty:
        summary = (
            ok_df.groupby(
                ["mode", "context_len", "new_tokens", "total_context_len"],
                as_index=False,
            )[["decode_ms_token_mean", "throughput_tok_s_mean"]]
            .mean()
            .sort_values("context_len")
        )
        print("\nAggregated summary")
        print(summary.to_string(index=False))

    artifacts_dir = get_artifacts_dir()
    correctness_path = os.path.join(artifacts_dir, "kv_cache_correctness.csv")
    results_path = os.path.join(artifacts_dir, "kv_cache_benchmark_results.csv")
    correctness_df.to_csv(correctness_path, index=False)
    df.to_csv(results_path, index=False)
    print(f"\nSaved: {correctness_path}")
    print(f"Saved: {results_path}")

    plot_sweep(df)
    plot_comparison(df)


if __name__ == "__main__":
    main()

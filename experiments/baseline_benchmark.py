import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from inference.generate import generate
from model.gpt import GPT, GPTConfig, load_hf_weights


DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_PROMPT = "Once upon a time"
DEFAULT_TOTAL_LENGTHS = [32, 128, 512, 768, 1024]
DEFAULT_SEED = 1234


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return [int(item) for item in items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark no-cache baseline generation across total context lengths."
        )
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name to load.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Fixed prompt used for every benchmark setting.",
    )
    parser.add_argument(
        "--total-lengths",
        type=parse_int_list,
        default=DEFAULT_TOTAL_LENGTHS,
        help="Comma-separated total prompt + generated lengths, for example 32,128,512,768,1024.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Warmup runs per total length before measurements.",
    )
    parser.add_argument(
        "--measure-runs",
        type=int,
        default=5,
        help="Measured runs per total length.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, for example cpu or cuda.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base seed for reproducible sampling.",
    )
    parser.add_argument(
        "--include-percentiles",
        action="store_true",
        help="Include p50 and p95 columns in the output CSV.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return 0.0


def get_artifacts_dir() -> Path:
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def build_model(model_name: str, device: str) -> GPT:
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_model.eval()

    config = GPTConfig(
        vocab_size=hf_model.config.vocab_size,
        block_size=hf_model.config.n_positions,
        n_layer=hf_model.config.n_layer,
        n_head=hf_model.config.n_head,
        n_embd=hf_model.config.n_embd,
    )
    model = GPT(config).to(device)
    load_hf_weights(model, hf_model)
    model.eval()
    return model


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    rank = (len(sorted_values) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}

    sorted_values = sorted(values)
    return {
        "mean": float(sum(values) / len(values)),
        "p50": percentile(sorted_values, 50.0),
        "p95": percentile(sorted_values, 95.0),
    }


@torch.inference_mode()
def run_generation(
    model: GPT,
    input_ids: torch.Tensor,
    total_length: int,
    seed: int,
) -> torch.Tensor:
    set_seed(seed)
    return generate(
        model,
        input_ids.clone(),
        use_kv_cache=False,
        max_context_len=total_length,
    )


def benchmark_total_length(
    model: GPT,
    input_ids: torch.Tensor,
    total_length: int,
    warmup_runs: int,
    measure_runs: int,
    device: str,
    seed: int,
    include_percentiles: bool,
) -> dict[str, float | int | str]:
    prompt_length = input_ids.shape[1]
    generated_tokens = total_length - prompt_length

    if generated_tokens < 0:
        return {
            "mode": "baseline",
            "prompt_length": prompt_length,
            "new_tokens": generated_tokens,
            "total_context_len": total_length,
            "status": "SKIPPED_PROMPT_TOO_LONG",
        }

    for run_idx in range(warmup_runs):
        _ = run_generation(
            model=model,
            input_ids=input_ids,
            total_length=total_length,
            seed=seed + run_idx,
        )

    timings = []
    peak_memories = []

    for run_idx in range(measure_runs):
        sync_if_needed(device)
        reset_peak_memory_if_needed(device)

        start = time.perf_counter()
        _ = run_generation(
            model=model,
            input_ids=input_ids,
            total_length=total_length,
            seed=seed + 1000 + run_idx,
        )
        sync_if_needed(device)
        end = time.perf_counter()

        timings.append(end - start)
        peak_memories.append(get_peak_memory_mb(device))

    latency_stats = summarize(timings)
    if generated_tokens > 0:
        latency_per_token = [
            latency * 1000.0 / generated_tokens for latency in timings
        ]
        throughput = [generated_tokens / latency for latency in timings]
    else:
        latency_per_token = [0.0 for _ in timings]
        throughput = [0.0 for _ in timings]

    token_latency_stats = summarize(latency_per_token)
    throughput_stats = summarize(throughput)
    memory_stats = summarize(peak_memories)

    row = {
        "mode": "baseline",
        "prompt_length": prompt_length,
        "new_tokens": generated_tokens,
        "total_context_len": total_length,
        "status": "OK",
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "e2e_latency_s_mean": latency_stats["mean"],
        "latency_ms_token_mean": token_latency_stats["mean"],
        "throughput_tok_s_mean": throughput_stats["mean"],
        "peak_gpu_memory_mb_mean": memory_stats["mean"],
    }

    if include_percentiles:
        row.update(
            {
                "e2e_latency_s_p50": latency_stats["p50"],
                "e2e_latency_s_p95": latency_stats["p95"],
                "latency_ms_token_p50": token_latency_stats["p50"],
                "latency_ms_token_p95": token_latency_stats["p95"],
                "throughput_tok_s_p50": throughput_stats["p50"],
                "throughput_tok_s_p95": throughput_stats["p95"],
                "peak_gpu_memory_mb_p50": memory_stats["p50"],
                "peak_gpu_memory_mb_p95": memory_stats["p95"],
            }
        )

    return row


def save_results(rows: list[dict[str, float | int | str]], artifacts_dir: Path) -> Path:
    csv_path = artifacts_dir / "baseline_benchmark_results.csv"
    fieldnames = sorted({field for row in rows for field in row.keys()})
    preferred = [
        "mode",
        "prompt_length",
        "new_tokens",
        "total_context_len",
        "status",
        "warmup_runs",
        "measure_runs",
        "e2e_latency_s_mean",
        "e2e_latency_s_p50",
        "e2e_latency_s_p95",
        "latency_ms_token_mean",
        "latency_ms_token_p50",
        "latency_ms_token_p95",
        "throughput_tok_s_mean",
        "throughput_tok_s_p50",
        "throughput_tok_s_p95",
        "peak_gpu_memory_mb_mean",
        "peak_gpu_memory_mb_p50",
        "peak_gpu_memory_mb_p95",
    ]
    ordered_fieldnames = [name for name in preferred if name in fieldnames]
    ordered_fieldnames.extend(name for name in fieldnames if name not in ordered_fieldnames)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def plot_results(rows: list[dict[str, float | int | str]], artifacts_dir: Path) -> Path | None:
    ok_rows = [
        row for row in rows
        if row.get("status") == "OK"
    ]
    if not ok_rows:
        print("No successful benchmark rows to plot.")
        return None

    ok_rows = sorted(ok_rows, key=lambda row: int(row["total_context_len"]))
    total_lengths = [int(row["total_context_len"]) for row in ok_rows]
    generated_tokens = [int(row["new_tokens"]) for row in ok_rows]
    e2e_latency = [float(row["e2e_latency_s_mean"]) for row in ok_rows]
    token_latency = [float(row["latency_ms_token_mean"]) for row in ok_rows]
    throughput = [float(row["throughput_tok_s_mean"]) for row in ok_rows]
    peak_memory = [float(row["peak_gpu_memory_mb_mean"]) for row in ok_rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    color = "#2563eb"
    grid_style = {"alpha": 0.25}

    plots = [
        (axes[0, 0], e2e_latency, "End-to-End Latency", "Seconds"),
        (axes[0, 1], token_latency, "Latency Per Generated Token", "ms/token"),
        (axes[1, 0], throughput, "Throughput", "tokens/sec"),
        (axes[1, 1], peak_memory, "Peak GPU Memory", "MB"),
    ]

    for ax, values, title, ylabel in plots:
        ax.plot(total_lengths, values, marker="o", linewidth=2, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, **grid_style)

    for ax in axes[1, :]:
        ax.set_xlabel("Total context length (prompt + generated tokens)")

    for idx, (total_length, new_tokens) in enumerate(zip(total_lengths, generated_tokens)):
        axes[0, 0].annotate(
            f"+{new_tokens}",
            xy=(total_length, e2e_latency[idx]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    prompt_length = int(ok_rows[0]["prompt_length"])
    fig.suptitle(
        f"Baseline generation benchmark (prompt={prompt_length} tokens)",
        fontsize=14,
    )
    fig.tight_layout()

    png_path = artifacts_dir / "baseline_benchmark_results.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if args.measure_runs <= 0:
        raise ValueError("--measure-runs must be positive")
    if any(total_length <= 0 for total_length in args.total_lengths):
        raise ValueError("--total-lengths must contain positive integers")

    artifacts_dir = get_artifacts_dir()
    set_seed(args.seed)

    print(f"Using device: {args.device}")
    print(f"Artifacts dir: {artifacts_dir}")
    print(f"Total lengths: {args.total_lengths}")

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = build_model(args.model_name, args.device)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
    prompt_length = input_ids.shape[1]

    print(f"Prompt length: {prompt_length}")
    print(f"Model block size: {model.config.block_size}")

    rows = []
    for total_length in args.total_lengths:
        if total_length > model.config.block_size:
            row = {
                "mode": "baseline",
                "prompt_length": prompt_length,
                "new_tokens": total_length - prompt_length,
                "total_context_len": total_length,
                "status": "SKIPPED_BLOCK_SIZE",
            }
        else:
            print(f"\nBenchmarking total_context_len={total_length}")
            row = benchmark_total_length(
                model=model,
                input_ids=input_ids,
                total_length=total_length,
                warmup_runs=args.warmup_runs,
                measure_runs=args.measure_runs,
                device=args.device,
                seed=args.seed + total_length,
                include_percentiles=args.include_percentiles,
            )

        rows.append(row)
        print(row)

    csv_path = save_results(rows, artifacts_dir)
    png_path = plot_results(rows, artifacts_dir)
    print(f"\nSaved benchmark results to: {csv_path}")
    if png_path:
        print(f"Saved benchmark plot to: {png_path}")


if __name__ == "__main__":
    main()

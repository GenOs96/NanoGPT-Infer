import argparse
import csv
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.generate import generate
from model.gpt import GPT, GPTConfig, load_hf_weights


VOCAB_SIZE = 50257
N_EMBED = 768
N_LAYER = 12
N_HEAD = 12


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(device: str) -> GPT:
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.eval()

    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBED,
        n_layer=N_LAYER,
        n_head=N_HEAD,
    )
    model = GPT(config).to(device)
    load_hf_weights(model, hf_model)
    model.eval()
    return model


def run_generation(
    model: GPT,
    input_ids: torch.Tensor,
    max_context_len: int,
    use_kv_cache: bool,
    seed: int,
) -> tuple[torch.Tensor, float]:
    set_seed(seed)
    start_time = time.time()
    output = generate(
        model,
        input_ids.clone(),
        use_kv_cache=use_kv_cache,
        max_context_len=max_context_len,
    )
    end_time = time.time()
    return output, end_time - start_time


def print_metrics(
    label: str,
    output: torch.Tensor,
    latency: float,
    generated_tokens: int,
    tokenizer: GPT2Tokenizer,
) -> None:
    print(label)
    print("Output:", tokenizer.decode(output[0], skip_special_tokens=True))
    print(f"End-to-end latency: {latency:.2f} seconds")
    print(f"Generated tokens: {generated_tokens}")
    if generated_tokens > 0:
        print(f"Decode ms per token: {latency / generated_tokens * 1000:.2f} ms/token")
        print(f"Throughput: {generated_tokens / latency:.2f} tokens/sec")
    else:
        print("Decode ms per token: n/a")
        print("Throughput: n/a")
    print()


def build_metric_rows(
    latency_no_cache: float,
    latency_cache: float,
    generated_tokens: int,
    max_context_len: int,
) -> list[dict[str, float | int | str]]:
    labels = ["Without KV cache", "With KV cache"]
    latencies = [latency_no_cache, latency_cache]

    if generated_tokens > 0:
        decode_ms = [latency / generated_tokens * 1000 for latency in latencies]
        throughputs = [generated_tokens / latency for latency in latencies]
    else:
        decode_ms = [0.0, 0.0]
        throughputs = [0.0, 0.0]

    rows = []
    for label, latency, decode_latency, throughput in zip(
        labels,
        latencies,
        decode_ms,
        throughputs,
    ):
        rows.append(
            {
                "mode": label,
                "generated_tokens": generated_tokens,
                "max_context_len": max_context_len,
                "end_to_end_latency_s": latency,
                "decode_latency_ms_per_token": decode_latency,
                "throughput_tokens_per_sec": throughput,
            }
        )

    return rows


def save_metric_rows(rows: list[dict[str, float | int | str]]) -> str:
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    csv_path = os.path.join(artifacts_dir, "kv_cache_simple_test_metrics.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def save_metrics_plot(rows: list[dict[str, float | int | str]]) -> str:
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    png_path = os.path.join(artifacts_dir, "kv_cache_simple_test_metrics.png")

    labels = [str(row["mode"]) for row in rows]
    latencies = [float(row["end_to_end_latency_s"]) for row in rows]
    decode_ms = [float(row["decode_latency_ms_per_token"]) for row in rows]
    throughputs = [float(row["throughput_tokens_per_sec"]) for row in rows]
    max_context_len = int(rows[0]["max_context_len"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["#d95f02", "#1b9e77"]
    metrics = [
        (latencies, "End-to-end latency", "Seconds"),
        (decode_ms, "Decode latency", "ms/token"),
        (throughputs, "Throughput", "tokens/sec"),
    ]

    for ax, (values, title, ylabel) in zip(axes, metrics):
        bars = ax.bar(labels, values, color=colors[: len(values)])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=10)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

    fig.suptitle(f"KV Cache Performance Comparison (total context={max_context_len})")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GPT inference with and without KV cache."
    )
    parser.add_argument(
        "--prompt",
        default="Once upon a time",
        help="Prompt used for generation.",
    )
    parser.add_argument(
        "--max-context-len",
        type=int,
        default=1024,
        help="Total context budget for prompt plus generated tokens.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=400,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, for example cpu or cuda.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = build_model(device)

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    generated_tokens = args.max_context_len - input_ids.shape[1]
    if generated_tokens < 0:
        raise ValueError(
            f"Prompt length {input_ids.shape[1]} exceeds "
            f"max_context_len={args.max_context_len}"
        )

    output_no_cache, latency_no_cache = run_generation(
        model=model,
        input_ids=input_ids,
        max_context_len=args.max_context_len,
        use_kv_cache=False,
        seed=args.seed,
    )
    output_cache, latency_cache = run_generation(
        model=model,
        input_ids=input_ids,
        max_context_len=args.max_context_len,
        use_kv_cache=True,
        seed=args.seed,
    )

    match = torch.equal(output_no_cache, output_cache)
    print(f"Outputs match: {match}")
    print()

    print_metrics(
        "Without KV cache:",
        output_no_cache,
        latency_no_cache,
        generated_tokens,
        tokenizer,
    )
    print_metrics(
        "With KV cache:",
        output_cache,
        latency_cache,
        generated_tokens,
        tokenizer,
    )
    metric_rows = build_metric_rows(
        latency_no_cache=latency_no_cache,
        latency_cache=latency_cache,
        generated_tokens=generated_tokens,
        max_context_len=args.max_context_len,
    )
    csv_path = save_metric_rows(metric_rows)
    png_path = save_metrics_plot(metric_rows)
    print(f"Saved comparison metrics to: {csv_path}")
    print(f"Saved comparison plot to: {png_path}")


if __name__ == "__main__":
    main()

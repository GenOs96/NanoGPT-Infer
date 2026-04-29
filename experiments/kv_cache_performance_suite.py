import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from inference.kv_cache import KVCache
from inference.sampler import sample_logits
from model.gpt import GPT, GPTConfig, load_hf_weights


DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_MAX_CONTEXT_LEN = 1024
DEFAULT_WARMUP_RUNS = 1
DEFAULT_MEASURE_RUNS = 3
DEFAULT_SEED = 1234
DEFAULT_DTYPE = torch.float32

FIXED_PROMPT_VARY_DECODE = [
    (128, 64),
    (128, 128),
    (128, 256),
    (128, 512),
]
FIXED_DECODE_VARY_PROMPT = [
    (64, 64),
    (128, 64),
    (512, 64),
    (768, 64),
]
PREFILL_DECODE_BREAKDOWN = [
    (128, 64),
    (512, 64),
    (768, 128),
]
BATCH_SWEEP = [
    (128, 64, 1),
    (128, 64, 2),
    (128, 64, 4),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compact KV-cache performance suite for T4-sized runs. "
            "All default prompt + generated token settings stay <= 1024."
        )
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name to load.",
    )
    parser.add_argument(
        "--max-context-len",
        type=int,
        default=DEFAULT_MAX_CONTEXT_LEN,
        help="Maximum allowed prompt + generated token length.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
        help="Warmup runs per measured configuration.",
    )
    parser.add_argument(
        "--measure-runs",
        type=int,
        default=DEFAULT_MEASURE_RUNS,
        help="Measured runs per configuration.",
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
        help="Base seed for reproducible random inputs and sampling.",
    )
    parser.add_argument(
        "--include-phase-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the prefill/decode phase breakdown test.",
    )
    parser.add_argument(
        "--include-batch-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the small batch-size sweep.",
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


def validate_config(prompt_len: int, generated_tokens: int, max_context_len: int) -> None:
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be positive, got {prompt_len}")
    if generated_tokens < 0:
        raise ValueError(f"generated_tokens must be non-negative, got {generated_tokens}")
    total_len = prompt_len + generated_tokens
    if total_len > max_context_len:
        raise ValueError(
            f"prompt_len + generated_tokens must be <= {max_context_len}, "
            f"got {prompt_len} + {generated_tokens} = {total_len}"
        )


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
    model = GPT(config)
    load_hf_weights(model, hf_model)
    model = model.to(device=device, dtype=DEFAULT_DTYPE)
    model.eval()
    return model


def make_input_ids(
    vocab_size: int,
    prompt_len: int,
    batch_size: int,
    device: str,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randint(
        0,
        vocab_size,
        (batch_size, prompt_len),
        device=device,
        generator=generator,
    )


def build_kv_cache(model: GPT, batch_size: int, total_len: int, device: str) -> KVCache:
    return KVCache(
        n_layer=model.config.n_layer,
        batch_size=batch_size,
        n_head=model.config.n_head,
        max_seq_len=total_len,
        head_dim=model.config.n_embd // model.config.n_head,
        device=device,
    )


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0}
    return {"mean": float(sum(values) / len(values))}


@torch.inference_mode()
def generate_no_cache(
    model: GPT,
    input_ids: torch.Tensor,
    generated_tokens: int,
) -> torch.Tensor:
    tokens = input_ids.clone()
    for _ in range(generated_tokens):
        logits = model(tokens)
        next_token = sample_logits(logits)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens


@torch.inference_mode()
def generate_with_kv_cache(
    model: GPT,
    input_ids: torch.Tensor,
    generated_tokens: int,
) -> torch.Tensor:
    tokens = input_ids.clone()
    batch_size, prompt_len = tokens.shape
    total_len = prompt_len + generated_tokens
    kv_cache = build_kv_cache(model, batch_size, total_len, str(tokens.device))
    logits = model(tokens, kv_cache=kv_cache)

    for step in range(generated_tokens):
        next_token = sample_logits(logits)
        tokens = torch.cat([tokens, next_token], dim=1)
        if step < generated_tokens - 1:
            logits = model(next_token, kv_cache=kv_cache)

    return tokens


def run_e2e_once(
    model: GPT,
    input_ids: torch.Tensor,
    generated_tokens: int,
    mode: str,
) -> torch.Tensor:
    if mode == "no_cache":
        return generate_no_cache(model, input_ids, generated_tokens)
    if mode == "kv_cache":
        return generate_with_kv_cache(model, input_ids, generated_tokens)
    raise ValueError(f"Unknown mode: {mode}")


def benchmark_e2e_config(
    test_name: str,
    model: GPT,
    input_ids: torch.Tensor,
    generated_tokens: int,
    mode: str,
    warmup_runs: int,
    measure_runs: int,
    device: str,
    seed: int,
) -> dict[str, float | int | str]:
    batch_size, prompt_len = input_ids.shape
    total_len = prompt_len + generated_tokens

    for run_idx in range(warmup_runs):
        set_seed(seed + run_idx)
        _ = run_e2e_once(model, input_ids, generated_tokens, mode)

    timings = []
    peak_memories = []
    for run_idx in range(measure_runs):
        set_seed(seed + 1000 + run_idx)
        sync_if_needed(device)
        reset_peak_memory_if_needed(device)
        start = time.perf_counter()
        _ = run_e2e_once(model, input_ids, generated_tokens, mode)
        sync_if_needed(device)
        end = time.perf_counter()
        timings.append(end - start)
        peak_memories.append(get_peak_memory_mb(device))

    latency_stats = summarize(timings)
    if generated_tokens > 0:
        ms_per_token = [latency * 1000.0 / generated_tokens for latency in timings]
        throughput = [(batch_size * generated_tokens) / latency for latency in timings]
    else:
        ms_per_token = [0.0 for _ in timings]
        throughput = [0.0 for _ in timings]

    return {
        "test_name": test_name,
        "mode": mode,
        "phase": "e2e",
        "prompt_len": prompt_len,
        "generated_tokens": generated_tokens,
        "total_context_len": total_len,
        "batch_size": batch_size,
        "measure_runs": measure_runs,
        "latency_s_mean": latency_stats["mean"],
        "ms_per_token_mean": summarize(ms_per_token)["mean"],
        "throughput_tok_s_mean": summarize(throughput)["mean"],
        "peak_gpu_memory_mb_mean": summarize(peak_memories)["mean"],
    }


@torch.inference_mode()
def run_phase_once(
    model: GPT,
    input_ids: torch.Tensor,
    generated_tokens: int,
    mode: str,
    device: str,
) -> tuple[float, float]:
    tokens = input_ids.clone()

    sync_if_needed(device)
    prefill_start = time.perf_counter()
    if mode == "kv_cache":
        batch_size, prompt_len = tokens.shape
        kv_cache = build_kv_cache(
            model,
            batch_size,
            prompt_len + generated_tokens,
            str(tokens.device),
        )
        logits = model(tokens, kv_cache=kv_cache)
    else:
        kv_cache = None
        logits = model(tokens)
    sync_if_needed(device)
    prefill_end = time.perf_counter()

    sync_if_needed(device)
    decode_start = time.perf_counter()
    for step in range(generated_tokens):
        next_token = sample_logits(logits)
        tokens = torch.cat([tokens, next_token], dim=1)
        if step < generated_tokens - 1:
            if mode == "kv_cache":
                logits = model(next_token, kv_cache=kv_cache)
            else:
                logits = model(tokens)
    sync_if_needed(device)
    decode_end = time.perf_counter()

    return prefill_end - prefill_start, decode_end - decode_start


def benchmark_phase_config(
    model: GPT,
    input_ids: torch.Tensor,
    generated_tokens: int,
    mode: str,
    warmup_runs: int,
    measure_runs: int,
    device: str,
    seed: int,
) -> list[dict[str, float | int | str]]:
    batch_size, prompt_len = input_ids.shape
    total_len = prompt_len + generated_tokens

    for run_idx in range(warmup_runs):
        set_seed(seed + run_idx)
        _ = run_phase_once(model, input_ids, generated_tokens, mode, device)

    prefill_timings = []
    decode_timings = []
    peak_memories = []
    for run_idx in range(measure_runs):
        set_seed(seed + 1000 + run_idx)
        reset_peak_memory_if_needed(device)
        prefill_latency, decode_latency = run_phase_once(
            model,
            input_ids,
            generated_tokens,
            mode,
            device,
        )
        prefill_timings.append(prefill_latency)
        decode_timings.append(decode_latency)
        peak_memories.append(get_peak_memory_mb(device))

    prefill_stats = summarize(prefill_timings)
    decode_stats = summarize(decode_timings)
    decode_ms_per_token = [
        latency * 1000.0 / generated_tokens for latency in decode_timings
    ] if generated_tokens > 0 else [0.0 for _ in decode_timings]
    decode_throughput = [
        (batch_size * generated_tokens) / latency for latency in decode_timings
    ] if generated_tokens > 0 else [0.0 for _ in decode_timings]

    common = {
        "test_name": "prefill_decode_breakdown",
        "mode": mode,
        "prompt_len": prompt_len,
        "generated_tokens": generated_tokens,
        "total_context_len": total_len,
        "batch_size": batch_size,
        "measure_runs": measure_runs,
        "peak_gpu_memory_mb_mean": summarize(peak_memories)["mean"],
    }
    return [
        {
            **common,
            "phase": "prefill",
            "latency_s_mean": prefill_stats["mean"],
            "ms_per_token_mean": prefill_stats["mean"] * 1000.0 / prompt_len,
            "throughput_tok_s_mean": batch_size * prompt_len / prefill_stats["mean"],
        },
        {
            **common,
            "phase": "decode",
            "latency_s_mean": decode_stats["mean"],
            "ms_per_token_mean": summarize(decode_ms_per_token)["mean"],
            "throughput_tok_s_mean": summarize(decode_throughput)["mean"],
        },
    ]


def run_e2e_suite(
    test_name: str,
    configs: list[tuple[int, int, int]],
    model: GPT,
    vocab_size: int,
    args: argparse.Namespace,
) -> list[dict[str, float | int | str]]:
    rows = []
    for prompt_len, generated_tokens, batch_size in configs:
        validate_config(prompt_len, generated_tokens, args.max_context_len)
        input_ids = make_input_ids(
            vocab_size,
            prompt_len,
            batch_size,
            args.device,
            args.seed + prompt_len + generated_tokens + batch_size,
        )
        for mode in ["no_cache", "kv_cache"]:
            print(
                f"{test_name}: mode={mode}, prompt={prompt_len}, "
                f"generated={generated_tokens}, batch={batch_size}"
            )
            rows.append(
                benchmark_e2e_config(
                    test_name=test_name,
                    model=model,
                    input_ids=input_ids,
                    generated_tokens=generated_tokens,
                    mode=mode,
                    warmup_runs=args.warmup_runs,
                    measure_runs=args.measure_runs,
                    device=args.device,
                    seed=args.seed + prompt_len * 10 + generated_tokens + batch_size,
                )
            )
    return rows


def run_phase_suite(
    model: GPT,
    vocab_size: int,
    args: argparse.Namespace,
) -> list[dict[str, float | int | str]]:
    rows = []
    for prompt_len, generated_tokens in PREFILL_DECODE_BREAKDOWN:
        validate_config(prompt_len, generated_tokens, args.max_context_len)
        input_ids = make_input_ids(
            vocab_size,
            prompt_len,
            1,
            args.device,
            args.seed + prompt_len + generated_tokens,
        )
        for mode in ["no_cache", "kv_cache"]:
            print(
                f"prefill_decode_breakdown: mode={mode}, prompt={prompt_len}, "
                f"generated={generated_tokens}, batch=1"
            )
            rows.extend(
                benchmark_phase_config(
                    model=model,
                    input_ids=input_ids,
                    generated_tokens=generated_tokens,
                    mode=mode,
                    warmup_runs=args.warmup_runs,
                    measure_runs=args.measure_runs,
                    device=args.device,
                    seed=args.seed + prompt_len * 10 + generated_tokens,
                )
            )
    return rows


def save_csv(rows: list[dict[str, float | int | str]], artifacts_dir: Path) -> Path:
    csv_path = artifacts_dir / "kv_cache_performance_suite.csv"
    preferred = [
        "test_name",
        "mode",
        "phase",
        "prompt_len",
        "generated_tokens",
        "total_context_len",
        "batch_size",
        "measure_runs",
        "latency_s_mean",
        "ms_per_token_mean",
        "throughput_tok_s_mean",
        "peak_gpu_memory_mb_mean",
    ]
    fieldnames = sorted({field for row in rows for field in row.keys()})
    ordered = [field for field in preferred if field in fieldnames]
    ordered.extend(field for field in fieldnames if field not in ordered)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def get_e2e_rows(rows: list[dict[str, float | int | str]], test_name: str):
    return [
        row for row in rows
        if row["test_name"] == test_name and row["phase"] == "e2e"
    ]


def plot_mode_lines(
    ax,
    rows: list[dict[str, float | int | str]],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    colors = {"no_cache": "#d95f02", "kv_cache": "#1b9e77"}
    for mode in ["no_cache", "kv_cache"]:
        mode_rows = sorted(
            [row for row in rows if row["mode"] == mode],
            key=lambda row: float(row[x_key]),
        )
        if not mode_rows:
            continue
        x_values = [float(row[x_key]) for row in mode_rows]
        y_values = [float(row[y_key]) for row in mode_rows]
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            color=colors[mode],
            label=mode,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def plot_phase_breakdown(ax, rows: list[dict[str, float | int | str]]) -> None:
    phase_rows = [
        row for row in rows
        if row["test_name"] == "prefill_decode_breakdown"
    ]
    if not phase_rows:
        ax.set_axis_off()
        ax.set_title("Prefill vs Decode Latency")
        ax.text(0.5, 0.5, "Phase test disabled", ha="center", va="center")
        return

    labels = []
    prefill_ms = []
    decode_ms = []
    ratios = []
    for prompt_len, generated_tokens in PREFILL_DECODE_BREAKDOWN:
        for mode in ["no_cache", "kv_cache"]:
            matching = [
                row for row in phase_rows
                if row["prompt_len"] == prompt_len
                and row["generated_tokens"] == generated_tokens
                and row["mode"] == mode
            ]
            prefill_row = next((row for row in matching if row["phase"] == "prefill"), None)
            decode_row = next((row for row in matching if row["phase"] == "decode"), None)
            if prefill_row and decode_row:
                labels.append(f"{mode}\np{prompt_len}/g{generated_tokens}")
                prefill_latency_ms = float(prefill_row["latency_s_mean"]) * 1000.0
                decode_latency_ms = float(decode_row["latency_s_mean"]) * 1000.0
                prefill_ms.append(prefill_latency_ms)
                decode_ms.append(decode_latency_ms)
                ratios.append(decode_latency_ms / prefill_latency_ms)

    x_values = list(range(len(labels)))
    width = 0.38
    ax.bar(
        [x - width / 2 for x in x_values],
        prefill_ms,
        width=width,
        color="#7570b3",
        label="prefill",
    )
    ax.bar(
        [x + width / 2 for x in x_values],
        decode_ms,
        width=width,
        color="#e7298a",
        label="decode",
    )
    for x, decode_latency_ms, ratio in zip(x_values, decode_ms, ratios):
        ax.text(
            x + width / 2,
            decode_latency_ms,
            f"{ratio:.0f}x",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_title("Prefill vs Decode Latency")
    ax.set_ylabel("Milliseconds (log scale)")
    ax.set_yscale("log")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)


def plot_results(rows: list[dict[str, float | int | str]], artifacts_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    fixed_prompt_rows = get_e2e_rows(rows, "fixed_prompt_vary_decode")
    fixed_decode_rows = get_e2e_rows(rows, "fixed_decode_vary_prompt")
    batch_rows = get_e2e_rows(rows, "batch_sweep")

    plot_mode_lines(
        axes[0, 0],
        fixed_prompt_rows,
        "generated_tokens",
        "ms_per_token_mean",
        "Fixed Prompt: Decode Latency",
        "Generated tokens",
        "ms/token",
    )
    plot_mode_lines(
        axes[0, 1],
        fixed_prompt_rows,
        "generated_tokens",
        "throughput_tok_s_mean",
        "Fixed Prompt: Throughput",
        "Generated tokens",
        "tokens/sec",
    )
    plot_mode_lines(
        axes[1, 0],
        fixed_decode_rows,
        "prompt_len",
        "ms_per_token_mean",
        "Fixed Decode: Prompt-Length Cost",
        "Prompt length",
        "ms/token",
    )
    plot_mode_lines(
        axes[1, 1],
        fixed_decode_rows,
        "prompt_len",
        "peak_gpu_memory_mb_mean",
        "Fixed Decode: GPU Memory",
        "Prompt length",
        "Peak MB",
    )
    plot_phase_breakdown(axes[2, 0], rows)
    plot_mode_lines(
        axes[2, 1],
        batch_rows,
        "batch_size",
        "throughput_tok_s_mean",
        "Batch Sweep: Throughput",
        "Batch size",
        "tokens/sec",
    )

    fig.suptitle("KV-cache Performance Suite", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    png_path = artifacts_dir / "kv_cache_performance_suite.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    if args.max_context_len > DEFAULT_MAX_CONTEXT_LEN:
        raise ValueError("--max-context-len cannot exceed 1024 for this suite")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if args.measure_runs <= 0:
        raise ValueError("--measure-runs must be positive")

    artifacts_dir = get_artifacts_dir()
    print(f"Device: {args.device}")
    print(f"Artifacts dir: {artifacts_dir}")
    print(f"Warmup runs: {args.warmup_runs}, measure runs: {args.measure_runs}")

    set_seed(args.seed)
    model = build_model(args.model_name, args.device)
    if args.max_context_len > model.config.block_size:
        raise ValueError(
            f"max_context_len={args.max_context_len} exceeds "
            f"model block_size={model.config.block_size}"
        )

    rows: list[dict[str, float | int | str]] = []
    rows.extend(
        run_e2e_suite(
            "fixed_prompt_vary_decode",
            [(prompt, gen, 1) for prompt, gen in FIXED_PROMPT_VARY_DECODE],
            model,
            model.config.vocab_size,
            args,
        )
    )
    rows.extend(
        run_e2e_suite(
            "fixed_decode_vary_prompt",
            [(prompt, gen, 1) for prompt, gen in FIXED_DECODE_VARY_PROMPT],
            model,
            model.config.vocab_size,
            args,
        )
    )
    if args.include_phase_test:
        rows.extend(run_phase_suite(model, model.config.vocab_size, args))
    if args.include_batch_test:
        rows.extend(
            run_e2e_suite(
                "batch_sweep",
                BATCH_SWEEP,
                model,
                model.config.vocab_size,
                args,
            )
        )

    csv_path = save_csv(rows, artifacts_dir)
    png_path = plot_results(rows, artifacts_dir)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()

import argparse
import csv
import os
import random
import shlex
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from inference.sampler import sample_logits
from model.gpt import GPT, GPTConfig, load_hf_weights


DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_SEED = 1234


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
    artifacts_dir = Path(__file__).resolve().parent / "artifacts" / "baseline_profiler"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


@contextmanager
def nvtx_range(name: str, enabled: bool):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


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


@torch.inference_mode()
def run_generation_workload(
    model: GPT,
    input_ids: torch.Tensor,
    max_context_len: int,
    enable_nvtx: bool,
    mark_decode_steps: bool,
) -> torch.Tensor:
    tokens = input_ids.clone()
    batch_size, prompt_length = tokens.shape
    max_new_tokens = max_context_len - prompt_length
    range_name = f"baseline_generate_b{batch_size}_s{prompt_length}_d{max_new_tokens}"

    with nvtx_range(range_name, enable_nvtx):
        with nvtx_range("decode", enable_nvtx):
            for step in range(max_new_tokens):
                step_name = f"decode_step_{step}" if mark_decode_steps else None
                with nvtx_range(step_name, enable_nvtx) if step_name else nullcontext():
                    logits = model(tokens)
                    next_token = sample_logits(logits)
                    tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a simple NanoGPT inference baseline with optional PyTorch "
            "Profiler traces and NVTX ranges for Nsight Systems."
        )
    )
    parser.add_argument(
        "--profile-tool",
        choices=["torch", "none"],
        default="torch",
        help=(
            "Use 'torch' for PyTorch Profiler traces. "
            "Use 'none' when wrapping the script with nsys or ncu."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name to load.",
    )
    parser.add_argument(
        "--prompt",
        default="Once upon a time",
        help="Prompt used for generation.",
    )
    parser.add_argument(
        "--num-new-tokens",
        type=int,
        default=16,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, for example cpu or cuda.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Warmup iterations before active profiling starts.",
    )
    parser.add_argument(
        "--active-steps",
        type=int,
        default=2,
        help="Active profiling iterations to record or summarize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base seed for reproducible sampling.",
    )
    parser.add_argument(
        "--mark-decode-steps",
        action="store_true",
        help="Emit per-token NVTX ranges inside the decode loop.",
    )
    parser.add_argument(
        "--disable-nvtx",
        action="store_true",
        help="Disable NVTX ranges even when CUDA is available.",
    )
    parser.add_argument(
        "--torch-row-limit",
        type=int,
        default=25,
        help="Number of rows to print in the PyTorch Profiler summary table.",
    )
    return parser.parse_args()


def run_timed_steps(
    model: GPT,
    input_ids: torch.Tensor,
    max_context_len: int,
    warmup_steps: int,
    active_steps: int,
    enable_nvtx: bool,
    mark_decode_steps: bool,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, list[float], list[float]]:
    total_steps = warmup_steps + active_steps
    step_latencies = []
    peak_memories = []
    output = input_ids

    for step in range(total_steps):
        set_seed(seed + step)
        sync_if_needed(device)
        reset_peak_memory_if_needed(device)
        start = time.perf_counter()
        output = run_generation_workload(
            model=model,
            input_ids=input_ids,
            max_context_len=max_context_len,
            enable_nvtx=enable_nvtx,
            mark_decode_steps=mark_decode_steps,
        )
        sync_if_needed(device)
        end = time.perf_counter()
        step_latencies.append(end - start)
        peak_memories.append(get_peak_memory_mb(device))

    return output, step_latencies[warmup_steps:], peak_memories[warmup_steps:]


def build_summary_row(
    profile_tool: str,
    input_ids: torch.Tensor,
    max_context_len: int,
    warmup_steps: int,
    active_steps: int,
    active_timings: list[float],
    active_peak_memories: list[float],
    trace_dir: Path | None,
    run_name: str,
) -> dict[str, float | int | str]:
    batch_size, prompt_length = input_ids.shape
    max_new_tokens = max_context_len - prompt_length
    latency_stats = summarize(active_timings)
    decode_ms = [
        latency * 1000.0 / max_new_tokens for latency in active_timings
    ] if max_new_tokens > 0 else [0.0 for _ in active_timings]
    throughput = [
        (batch_size * max_new_tokens) / latency for latency in active_timings
    ] if max_new_tokens > 0 else [0.0 for _ in active_timings]

    return {
        "profile_tool": profile_tool,
        "batch_size": batch_size,
        "prompt_length": prompt_length,
        "max_new_tokens": max_new_tokens,
        "total_context_len": max_context_len,
        "warmup_steps": warmup_steps,
        "active_steps": active_steps,
        "latency_mean_s": latency_stats["mean"],
        "latency_min_s": latency_stats["min"],
        "latency_max_s": latency_stats["max"],
        "decode_latency_mean_ms_per_token": summarize(decode_ms)["mean"],
        "throughput_mean_tokens_per_sec": summarize(throughput)["mean"],
        "peak_cuda_memory_mb": summarize(active_peak_memories)["max"],
        "trace_dir": str(trace_dir) if trace_dir else "",
        "run_name": run_name,
    }


def run_torch_profile(
    model: GPT,
    input_ids: torch.Tensor,
    max_context_len: int,
    warmup_steps: int,
    active_steps: int,
    enable_nvtx: bool,
    mark_decode_steps: bool,
    device: str,
    seed: int,
    row_limit: int,
    artifacts_dir: Path,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    trace_dir = artifacts_dir / "torch_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda") and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    batch_size, prompt_length = input_ids.shape
    max_new_tokens = max_context_len - prompt_length
    run_name = f"simple_baseline_b{batch_size}_s{prompt_length}_d{max_new_tokens}_{device.replace(':', '_')}"
    total_steps = warmup_steps + active_steps
    timed_steps = []
    peak_memories = []
    output = input_ids

    with profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir), worker_name=run_name),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for step in range(total_steps):
            set_seed(seed + step)
            sync_if_needed(device)
            reset_peak_memory_if_needed(device)
            start = time.perf_counter()
            output = run_generation_workload(
                model=model,
                input_ids=input_ids,
                max_context_len=max_context_len,
                enable_nvtx=enable_nvtx,
                mark_decode_steps=mark_decode_steps,
            )
            sync_if_needed(device)
            end = time.perf_counter()
            prof.step()
            timed_steps.append(end - start)
            peak_memories.append(get_peak_memory_mb(device))

    sort_key = "cuda_time_total" if device.startswith("cuda") and torch.cuda.is_available() else "cpu_time_total"
    print(f"\nPyTorch Profiler summary for {run_name}")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=row_limit))

    row = build_summary_row(
        profile_tool="torch",
        input_ids=input_ids,
        max_context_len=max_context_len,
        warmup_steps=warmup_steps,
        active_steps=active_steps,
        active_timings=timed_steps[warmup_steps:],
        active_peak_memories=peak_memories[warmup_steps:],
        trace_dir=trace_dir,
        run_name=run_name,
    )
    return output, row


def run_unprofiled(
    model: GPT,
    input_ids: torch.Tensor,
    max_context_len: int,
    warmup_steps: int,
    active_steps: int,
    enable_nvtx: bool,
    mark_decode_steps: bool,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    output, active_timings, active_peak_memories = run_timed_steps(
        model=model,
        input_ids=input_ids,
        max_context_len=max_context_len,
        warmup_steps=warmup_steps,
        active_steps=active_steps,
        enable_nvtx=enable_nvtx,
        mark_decode_steps=mark_decode_steps,
        device=device,
        seed=seed,
    )
    batch_size, prompt_length = input_ids.shape
    max_new_tokens = max_context_len - prompt_length
    run_name = f"simple_baseline_b{batch_size}_s{prompt_length}_d{max_new_tokens}"
    row = build_summary_row(
        profile_tool="none",
        input_ids=input_ids,
        max_context_len=max_context_len,
        warmup_steps=warmup_steps,
        active_steps=active_steps,
        active_timings=active_timings,
        active_peak_memories=active_peak_memories,
        trace_dir=None,
        run_name=run_name,
    )
    return output, row


def save_summary_csv(row: dict[str, float | int | str], profile_tool: str, artifacts_dir: Path) -> Path:
    csv_path = artifacts_dir / f"baseline_profiler_summary_{profile_tool}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return csv_path


def print_nsight_commands(args: argparse.Namespace) -> None:
    base_cmd = (
        "python experiments/baseline_profiler.py "
        f"--profile-tool none "
        f"--model-name {shlex.quote(args.model_name)} "
        f"--prompt {shlex.quote(args.prompt)} "
        f"--num-new-tokens {args.num_new_tokens} "
        f"--device {shlex.quote(args.device)} "
        f"--warmup-steps {args.warmup_steps} "
        f"--active-steps {args.active_steps} "
        f"--seed {args.seed}"
    )
    if args.mark_decode_steps:
        base_cmd += " --mark-decode-steps"
    if args.disable_nvtx:
        base_cmd += " --disable-nvtx"

    nsys_cmd = (
        "nsys profile --trace=cuda,nvtx,osrt "
        "--sample=none "
        "--output experiments/artifacts/baseline_profiler/nsys_baseline_timeline "
        + base_cmd
    )
    ncu_cmd = (
        "ncu --set full "
        "--nvtx "
        "--nvtx-include decode_step "
        "--export experiments/artifacts/baseline_profiler/ncu_baseline_decode "
        + base_cmd
    )

    print("\nNsight Systems command")
    print(nsys_cmd)
    print("\nNsight Compute command")
    print(ncu_cmd)


def main() -> None:
    args = parse_args()
    if args.num_new_tokens < 0:
        raise ValueError("--num-new-tokens must be non-negative")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.active_steps <= 0:
        raise ValueError("--active-steps must be positive")

    artifacts_dir = get_artifacts_dir()
    device = args.device
    enable_nvtx = not args.disable_nvtx and device.startswith("cuda") and torch.cuda.is_available()

    print(f"Using device: {device}")
    print(f"Artifacts dir: {artifacts_dir}")
    print(f"Profile tool: {args.profile_tool}")
    print(f"Warmup steps: {args.warmup_steps}, active steps: {args.active_steps}")

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = build_model(args.model_name, device)

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    max_context_len = input_ids.shape[1] + args.num_new_tokens
    if max_context_len > model.config.block_size:
        raise ValueError(
            f"prompt length + num_new_tokens exceeds block size: "
            f"{input_ids.shape[1]} + {args.num_new_tokens} > {model.config.block_size}"
        )

    if args.profile_tool == "torch":
        output, row = run_torch_profile(
            model=model,
            input_ids=input_ids,
            max_context_len=max_context_len,
            warmup_steps=args.warmup_steps,
            active_steps=args.active_steps,
            enable_nvtx=enable_nvtx,
            mark_decode_steps=args.mark_decode_steps,
            device=device,
            seed=args.seed,
            row_limit=args.torch_row_limit,
            artifacts_dir=artifacts_dir,
        )
    else:
        output, row = run_unprofiled(
            model=model,
            input_ids=input_ids,
            max_context_len=max_context_len,
            warmup_steps=args.warmup_steps,
            active_steps=args.active_steps,
            enable_nvtx=enable_nvtx,
            mark_decode_steps=args.mark_decode_steps,
            device=device,
            seed=args.seed,
        )

    print("\nGenerated text:", tokenizer.decode(output[0], skip_special_tokens=True))
    print(row)
    summary_csv = save_summary_csv(row, args.profile_tool, artifacts_dir)
    print(f"Saved summary CSV: {summary_csv}")
    if args.profile_tool == "torch":
        print(f"TensorBoard logdir: {artifacts_dir / 'torch_traces'}")
    print_nsight_commands(args)


if __name__ == "__main__":
    main()

import argparse
import os
import random
import sys
import time
from typing import Callable

import torch
from transformers import GPT2LMHeadModel


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model.gpt import GPT, GPTConfig, load_hf_weights


DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_DTYPE = torch.float32
DEFAULT_SEED = 1234

KVCacheModel = Callable[..., tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]]
PastKV = list[tuple[torch.Tensor, torch.Tensor]] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small baseline vs KV-cache comparison for quick iteration."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=512)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the optimized KV-cache model with torch.compile.",
    )
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        help="torch.compile mode for the optimized path.",
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
    batch_size: int,
    prompt_len: int,
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


def next_token_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def append_past_kv(
    past_kv: PastKV,
    kv_updates: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if past_kv is None:
        return kv_updates
    return [
        (
            torch.cat([past_k, k_update], dim=2),
            torch.cat([past_v, v_update], dim=2),
        )
        for (past_k, past_v), (k_update, v_update) in zip(past_kv, kv_updates)
    ]


@torch.inference_mode()
def generate_baseline(
    model: GPT,
    input_ids: torch.Tensor,
    new_tokens: int,
) -> torch.Tensor:
    tokens = input_ids.clone()
    for _ in range(new_tokens):
        logits = model(tokens)
        next_token = next_token_from_logits(logits)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens


@torch.inference_mode()
def generate_optimized(
    model: GPT,
    kv_cache_model: KVCacheModel,
    input_ids: torch.Tensor,
    new_tokens: int,
) -> torch.Tensor:
    tokens = input_ids.clone()
    _, prompt_len = tokens.shape
    past_kv: PastKV = None
    start_pos = 0

    logits, kv_updates = kv_cache_model(tokens, past_kv, start_pos)
    past_kv = append_past_kv(past_kv, kv_updates)
    start_pos += prompt_len
    for step in range(new_tokens):
        next_token = next_token_from_logits(logits)
        tokens = torch.cat([tokens, next_token], dim=1)
        if step < new_tokens - 1:
            logits, kv_updates = kv_cache_model(next_token, past_kv, start_pos)
            past_kv = append_past_kv(past_kv, kv_updates)
            start_pos += 1
    return tokens


def measure(
    name: str,
    fn: Callable[[], torch.Tensor],
    batch_size: int,
    new_tokens: int,
    warmup_runs: int,
    measure_runs: int,
    device: str,
) -> dict[str, float | str]:
    for _ in range(warmup_runs):
        _ = fn()
    sync_if_needed(device)

    timings = []
    for _ in range(measure_runs):
        sync_if_needed(device)
        start = time.perf_counter()
        _ = fn()
        sync_if_needed(device)
        timings.append(time.perf_counter() - start)

    latency_s = sum(timings) / len(timings)
    generated = batch_size * new_tokens
    throughput = generated / latency_s if latency_s > 0 else 0.0
    return {
        "name": name,
        "latency_ms": latency_s * 1000.0,
        "ms_per_token": latency_s * 1000.0 / new_tokens if new_tokens > 0 else 0.0,
        "tokens_per_s": throughput,
    }


def print_result(result: dict[str, float | str]) -> None:
    print(
        f"{result['name']:>10} | "
        f"latency={result['latency_ms']:9.2f} ms | "
        f"ms/token={result['ms_per_token']:8.3f} | "
        f"throughput={result['tokens_per_s']:9.2f} tok/s"
    )


def main() -> None:
    args = parse_args()
    if args.prompt_len <= 0:
        raise ValueError("--prompt-len must be positive")
    if args.new_tokens <= 0:
        raise ValueError("--new-tokens must be positive")
    if args.measure_runs <= 0:
        raise ValueError("--measure-runs must be positive")

    set_seed(args.seed)
    model = build_model(args.model_name, args.device)
    if args.prompt_len + args.new_tokens > model.config.block_size:
        raise ValueError(
            f"prompt_len + new_tokens must be <= {model.config.block_size}, "
            f"got {args.prompt_len} + {args.new_tokens}"
        )

    input_ids = make_input_ids(
        model.config.vocab_size,
        args.batch_size,
        args.prompt_len,
        args.device,
        args.seed,
    )
    kv_cache_model = (
        torch.compile(model.forward_with_past, mode=args.compile_mode)
        if args.compile
        else model.forward_with_past
    )

    print(
        f"device={args.device}, batch={args.batch_size}, "
        f"prompt={args.prompt_len}, new_tokens={args.new_tokens}, "
        f"compile={args.compile}"
    )

    baseline = measure(
        "baseline",
        lambda: generate_baseline(model, input_ids, args.new_tokens),
        args.batch_size,
        args.new_tokens,
        args.warmup_runs,
        args.measure_runs,
        args.device,
    )
    optimized = measure(
        "optimized",
        lambda: generate_optimized(model, kv_cache_model, input_ids, args.new_tokens),
        args.batch_size,
        args.new_tokens,
        args.warmup_runs,
        args.measure_runs,
        args.device,
    )

    print_result(baseline)
    print_result(optimized)
    speedup = baseline["latency_ms"] / optimized["latency_ms"]
    print(f"   speedup | {speedup:.2f}x lower latency")


if __name__ == "__main__":
    main()

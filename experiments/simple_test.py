import argparse
import os
import sys
import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.generate import generate
from model.gpt import GPT, GPTConfig, load_hf_weights


VOCAB_SIZE = 50257
N_EMBED = 768
N_LAYER = 12
N_HEAD = 12


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple NanoGPT inference test."
    )
    parser.add_argument(
        "--prompt",
        default="Once upon a time",
        help="Prompt used for generation.",
    )
    parser.add_argument(
        "--num-new-tokens",
        type=int,
        default=1,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, for example cpu or cuda.",
    )
    parser.add_argument(
        "--use-kv-cache",
        action="store_true",
        help="Use KV cache during generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = build_model(device)

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    max_context_len = input_ids.shape[1] + args.num_new_tokens

    start_time = time.time()
    output = generate(
        model,
        input_ids,
        use_kv_cache=args.use_kv_cache,
        max_context_len=max_context_len,
    )
    end_time = time.time()

    print("Generated text:", tokenizer.decode(output[0], skip_special_tokens=True))
    print(f"Generation time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

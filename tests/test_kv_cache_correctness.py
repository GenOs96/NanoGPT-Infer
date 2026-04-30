import random
import unittest

import torch

from inference.generate import generate
from inference.kv_cache import KVCache
from model.gpt import GPT, GPTConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_tiny_model() -> GPT:
    set_seed(1234)
    model = GPT(
        GPTConfig(
            vocab_size=32,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=16,
        )
    )
    model.eval()
    return model


def build_kv_cache(model: GPT, batch_size: int, max_seq_len: int) -> KVCache:
    return KVCache(
        n_layer=model.config.n_layer,
        batch_size=batch_size,
        n_head=model.config.n_head,
        max_seq_len=max_seq_len,
        head_dim=model.config.n_embd // model.config.n_head,
        device="cpu",
    )


class KVCacheCorrectnessTest(unittest.TestCase):
    def test_cached_logits_match_full_context_logits(self) -> None:
        model = build_tiny_model()
        prompt = torch.tensor([[1, 5, 7, 9]], dtype=torch.long)
        decode_tokens = torch.tensor([[3, 4, 2]], dtype=torch.long)
        kv_cache = build_kv_cache(
            model=model,
            batch_size=prompt.shape[0],
            max_seq_len=prompt.shape[1] + decode_tokens.shape[1],
        )

        with torch.inference_mode():
            full_prompt_logits = model(prompt)
            cached_logits = model(prompt, kv_cache=kv_cache)

            torch.testing.assert_close(
                cached_logits,
                full_prompt_logits,
                rtol=1e-5,
                atol=1e-5,
            )

            full_context = prompt
            for token_idx in range(decode_tokens.shape[1]):
                next_token = decode_tokens[:, token_idx : token_idx + 1]
                full_context = torch.cat([full_context, next_token], dim=1)

                full_logits = model(full_context)
                cached_logits = model(next_token, kv_cache=kv_cache)

                torch.testing.assert_close(
                    cached_logits[:, -1, :],
                    full_logits[:, -1, :],
                    rtol=1e-5,
                    atol=1e-5,
                )

        self.assertEqual(kv_cache.current_seq_len, full_context.shape[1])

    def test_cached_generation_matches_no_cache_generation(self) -> None:
        model = build_tiny_model()
        prompt = torch.tensor([[2, 6, 10, 14]], dtype=torch.long)
        max_context_len = 9

        set_seed(2026)
        no_cache_tokens = generate(
            model=model,
            idx=prompt.clone(),
            use_kv_cache=False,
            max_context_len=max_context_len,
        )

        set_seed(2026)
        cached_tokens = generate(
            model=model,
            idx=prompt.clone(),
            use_kv_cache=True,
            max_context_len=max_context_len,
        )

        self.assertTrue(torch.equal(cached_tokens, no_cache_tokens))
        self.assertEqual(cached_tokens.shape, (1, max_context_len))


if __name__ == "__main__":
    unittest.main()

import torch
from inference.sampler import sample_logits
from inference.kv_cache import KVCache

@torch.no_grad()
def generate(model, idx, use_kv_cache=False, max_context_len=1024):
    model.eval()
    B, T = idx.shape
    block_size = model.config.block_size

    if max_context_len > block_size:
        raise ValueError(
            f"max_context_len={max_context_len} exceeds model block_size={block_size}"
        )
    if T > max_context_len:
        raise ValueError(
            f"Input prompt length {T} exceeds max_context_len={max_context_len}"
        )

    max_new_tokens = max_context_len - T

    # KV cache is used to store past keys and values for each layer to speed up autoregressive decoding
    if use_kv_cache:
        kv_cache = KVCache(
            n_layer=model.config.n_layer,
            batch_size=B,
            n_head=model.config.n_head,
            # Use a fixed total context budget for prompt + generated tokens.
            max_seq_len=max_context_len,
            head_dim=model.config.n_embd // model.config.n_head,
            device=idx.device
        )

        # Prefill once on the full prompt.
        logits = model(idx, kv_cache=kv_cache)
        for _ in range(max_new_tokens):
            next_token = sample_logits(logits)
            idx = torch.cat([idx, next_token], dim=1)
            # Decode incrementally: feed only the new token.
            logits = model(next_token, kv_cache=kv_cache)
    else:
        kv_cache = None
        for _ in range(max_new_tokens):
            logits = model(idx, kv_cache=kv_cache)
            next_token = sample_logits(logits)
            idx = torch.cat([idx, next_token], dim=1)

    return idx

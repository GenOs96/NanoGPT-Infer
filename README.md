# NanoGPT-Infer
- A Minimal High-Performance GPT Inference Engine.

__Bare Bones components:__
- Token + positional embeddings
- Multi-head causal attention
- Transformer blocks
- Sampling-based generation

__Future Features:__
- KV cache v1:
    - Separate prefill and decode stages
    - Cache K/V tensors during prefill
    - Simple memory layout k_cache/v_cache:
        - Index dimensions: (num_layers)
        - Value dimensions: (batch, token_position, num_heads, head_dim)
        - Static preallocated KV cache based on max_tokens
    - Drawbacks:
        - Better locality, but GPU warps can still incur strided access
        - Static allocation may waste VRAM and complicate batching        

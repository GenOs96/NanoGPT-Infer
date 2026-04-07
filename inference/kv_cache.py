import torch

class KVCache:
    def __init__(self, n_layer, batch_size, n_head, max_seq_len, head_dim, device):

        self.k_cache = [
            torch.zeros(batch_size, max_seq_len, n_head, head_dim, device=device)
            for _ in range(n_layer)
        ]
        self.v_cache = [
            torch.zeros(batch_size, max_seq_len, n_head, head_dim, device=device)
            for _ in range(n_layer)
        ]
        self.current_seq_len = 0

    def update(self, layer_idx, k_update, v_update):

        if k_update.dim() == 3:
            k_update = k_update.unsqueeze(0)
        if v_update.dim() == 3:
            v_update = v_update.unsqueeze(0)

        if k_update.dim() != 4 or v_update.dim() != 4:
            raise ValueError(
                f"KV updates must be 4D [B,T,n_head,head_dim], got "
                f"k={tuple(k_update.shape)}, v={tuple(v_update.shape)}"
            )

        B, T, n_head, head_dim = k_update.shape  
        cache_B, cache_max_t, cache_n_head, cache_head_dim = self.k_cache[layer_idx].shape
        if B != cache_B or n_head != cache_n_head or head_dim != cache_head_dim:
            raise ValueError(
                f"KV shape mismatch with cache: update={tuple(k_update.shape)} "
                f"cache={tuple(self.k_cache[layer_idx].shape)}"
            )

        if layer_idx == 0:
            start = self.current_seq_len
            end = start + T
        else:
            end = self.current_seq_len
            start = end - T

        if start < 0:
            raise ValueError(
                f"Invalid KV write range for layer {layer_idx}: start={start}, end={end}, T={T}"
            )

        if end > cache_max_t:
            raise ValueError(
                f"KV cache overflow: trying to write up to {end}, "
                f"but max_seq_len={cache_max_t}"
            )

        self.k_cache[layer_idx][:, start:end, :, :] = k_update
        self.v_cache[layer_idx][:, start:end, :, :] = v_update

        if layer_idx == 0: 
            self.current_seq_len = end

        k = self.k_cache[layer_idx][:, :self.current_seq_len, :, :]
        v = self.v_cache[layer_idx][:, :self.current_seq_len, :, :]
        return k, v

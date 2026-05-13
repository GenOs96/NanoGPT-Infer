import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function


# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
class GPTConfig:
    def __init__(
        self,
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd


# -----------------------------------------------------------
# Attention Module
# -----------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(
        self,
        x,
        kv_cache=None,
        layer_idx=None,
        past_k=None,
        past_v=None,
        attn_mask=None,
        return_kv_update=False,
    ):
        B, T, C = x.shape

        attn_scope = (
            f"block_{layer_idx}.attention"
            if layer_idx is not None
            else "attention"
        )

        with record_function(attn_scope):
            B, T, C = x.shape
            is_decoding = False
            
            qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
            q, k, v = qkv.unbind(dim=2)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            k_update = k
            v_update = v

            if past_k is not None and past_v is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            if kv_cache is not None:
                with record_function("kv_cache"):
                    k, v = kv_cache.update(layer_idx, k, v)
                if q.size(2) == 1:
                    is_decoding = True
            
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=attn_mask is None and not is_decoding
            )

            y = y.transpose(1, 2).contiguous().view(B, T, C)

            y = self.proj(y)
            if return_kv_update:
                return y, (k_update, v_update)
            return y


# -----------------------------------------------------------
# MLP
# -----------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x, layer_idx=None):
        mlp_scope = f"block_{layer_idx}.mlp" if layer_idx is not None else "mlp"
        with record_function(mlp_scope):
            x = F.gelu(self.fc1(x))
            return self.fc2(x)


# -----------------------------------------------------------
# Transformer Block
# -----------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None, layer_idx=None):
        block_scope = f"block_{layer_idx}" if layer_idx is not None else "block"
        with record_function(block_scope):
            x = x + self.attn(
                self.ln1(x),
                kv_cache=kv_cache,
                layer_idx=layer_idx,
            )
            x = x + self.mlp(self.ln2(x), layer_idx=layer_idx)
            return x

    def forward_with_past(
        self,
        x,
        past_k=None,
        past_v=None,
        attn_mask=None,
        layer_idx=None,
    ):
        block_scope = f"block_{layer_idx}" if layer_idx is not None else "block"
        with record_function(block_scope):
            attn_out, kv_update = self.attn(
                self.ln1(x),
                layer_idx=layer_idx,
                past_k=past_k,
                past_v=past_v,
                attn_mask=attn_mask,
                return_kv_update=True,
            )
            x = x + attn_out
            x = x + self.mlp(self.ln2(x), layer_idx=layer_idx)
            return x, kv_update


# -----------------------------------------------------------
# GPT Model
# -----------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, kv_cache=None):
        B, T = idx.shape

        with record_function("gpt.forward"):
            if kv_cache is not None:
                start_pos = kv_cache.current_seq_len
                positions = torch.arange(
                    start_pos,
                    start_pos + T,
                    device=idx.device,
                ).unsqueeze(0)
            else:
                positions = torch.arange(0, T, device=idx.device).unsqueeze(0)

            x = self.token_emb(idx) + self.pos_emb(positions)

            for i, block in enumerate(self.blocks):
                x = block(x, kv_cache=kv_cache, layer_idx=i)

            x = self.ln_f(x)

            logits = self.lm_head(x)

            return logits

    def build_attention_mask(self, query_start, query_len, key_len, device):
        query_positions = torch.arange(
            query_start,
            query_start + query_len,
            device=device,
        ).unsqueeze(1)
        key_positions = torch.arange(key_len, device=device).unsqueeze(0)
        return (key_positions <= query_positions).view(1, 1, query_len, key_len)

    def forward_with_past(self, idx, past_kv=None, start_pos=0):
        _, T = idx.shape

        with record_function("gpt.forward"):
            positions = torch.arange(
                start_pos,
                start_pos + T,
                device=idx.device,
            ).unsqueeze(0)
            x = self.token_emb(idx) + self.pos_emb(positions)

            if past_kv is None:
                past_kv = [None for _ in range(len(self.blocks))]

            present_kv = []
            key_len = start_pos + T
            attn_mask = self.build_attention_mask(
                query_start=start_pos,
                query_len=T,
                key_len=key_len,
                device=idx.device,
            )

            for i, block in enumerate(self.blocks):
                layer_past = past_kv[i]
                past_k = layer_past[0] if layer_past is not None else None
                past_v = layer_past[1] if layer_past is not None else None
                x, kv_update = block.forward_with_past(
                    x,
                    past_k=past_k,
                    past_v=past_v,
                    attn_mask=attn_mask,
                    layer_idx=i,
                )
                present_kv.append(kv_update)

            x = self.ln_f(x)
            logits = self.lm_head(x)
            return logits, present_kv

# -----------------------------------------------------------
# Weight Loading (Hugging Face GPT-2)
# -----------------------------------------------------------
def load_hf_weights(model, hf_model):
    """
    Load weights from HuggingFace GPT2LMHeadModel into this model.
    """
    sd = hf_model.state_dict()

    with torch.no_grad():
        # embeddings
        model.token_emb.weight.copy_(sd["transformer.wte.weight"])
        model.pos_emb.weight.copy_(sd["transformer.wpe.weight"])

        # blocks
        for i, block in enumerate(model.blocks):
            prefix = f"transformer.h.{i}."

            # LayerNorms
            block.ln1.weight.copy_(sd[prefix + "ln_1.weight"])
            block.ln1.bias.copy_(sd[prefix + "ln_1.bias"])
            block.ln2.weight.copy_(sd[prefix + "ln_2.weight"])
            block.ln2.bias.copy_(sd[prefix + "ln_2.bias"])

            # Attention
            qkv_weight = sd[prefix + "attn.c_attn.weight"]
            qkv_bias = sd[prefix + "attn.c_attn.bias"]

            block.attn.qkv.weight.copy_(qkv_weight.t())
            block.attn.qkv.bias.copy_(qkv_bias)

            block.attn.proj.weight.copy_(sd[prefix + "attn.c_proj.weight"].t())
            block.attn.proj.bias.copy_(sd[prefix + "attn.c_proj.bias"])

            # MLP
            block.mlp.fc1.weight.copy_(sd[prefix + "mlp.c_fc.weight"].t())
            block.mlp.fc1.bias.copy_(sd[prefix + "mlp.c_fc.bias"])
            block.mlp.fc2.weight.copy_(sd[prefix + "mlp.c_proj.weight"].t())
            block.mlp.fc2.bias.copy_(sd[prefix + "mlp.c_proj.bias"])

        # final layer norm
        model.ln_f.weight.copy_(sd["transformer.ln_f.weight"])
        model.ln_f.bias.copy_(sd["transformer.ln_f.bias"])

        # lm head
        model.lm_head.weight.copy_(sd["lm_head.weight"])

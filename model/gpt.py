import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x, kv_cache=None, layer_idx=None):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        # Always move q/k/v into (B, T, n_head, head_dim) first so both
        # cached and non-cached paths share identical tensor layouts.
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # reshape
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # attention
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if kv_cache is None:
            # training / no cache
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        else:
            # cache mode: source length can exceed query length
            S = k.size(-2)
            q_start = S - T
            att = att.masked_fill(
                self.mask[:, :, q_start:S, :S] == 0,
                float("-inf"),
            )

        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)


# -----------------------------------------------------------
# MLP
# -----------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


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
        x = x + self.attn(self.ln1(x), kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + self.mlp(self.ln2(x))
        return x


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

        if kv_cache is not None:
            start_pos = kv_cache.current_seq_len
            positions = torch.arange(start_pos, start_pos + T, device=idx.device).unsqueeze(0)
        else:
            positions = torch.arange(0, T, device=idx.device).unsqueeze(0)

        x = self.token_emb(idx) + self.pos_emb(positions)

        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=kv_cache, layer_idx=i)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits

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

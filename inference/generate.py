import torch
from inference.sampler import sample_logits

@torch.no_grad()
def generate(model, idx, max_new_tokens=50):
    for _ in range(max_new_tokens):
        logits = model(idx)
        next_token = sample_logits(logits)
        idx = torch.cat([idx, next_token], dim=1)
    return idx
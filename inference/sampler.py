import torch

def sample_logits(logits, temperature=1.0, top_k=None):
    logits = logits[:, -1, :] / temperature

    if top_k:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
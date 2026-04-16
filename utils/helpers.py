import random
import torch
from typing import Dict, List
from models.next_word_model import NextWordModel
from utils.config import UNK_TOKEN, PAD_TOKEN

def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def predict_next_word(seed_words:  List[str],
                      model:       NextWordModel,
                      word2idx:    Dict[str, int],
                      idx2word:    Dict[int, str],
                      device:      torch.device,
                      max_seq_len: int,
                      top_k:       int = 5) -> str:
    """
    Given seed_words, return the model's top-1 predicted next word.
    Also prints a top-k probability table for inspection.
    """
    model.eval()
    unk_idx = word2idx[UNK_TOKEN]
    pad_idx = word2idx[PAD_TOKEN]

    indices = [word2idx.get(w, unk_idx) for w in seed_words]
    seq_len = min(len(indices), max_seq_len)
    padded  = [pad_idx] * max_seq_len
    padded[:seq_len] = indices[:seq_len]

    x      = torch.tensor([padded], dtype=torch.long, device=device)
    logits, _ = model(x)
    probs     = torch.softmax(logits[0], dim=-1)

    topk_probs, topk_idxs = torch.topk(probs, top_k)
    print(f"\n  [{model.rnn_type}]  Seed: {seed_words}")
    for rank, (p, i) in enumerate(zip(topk_probs.tolist(),
                                      topk_idxs.tolist()), 1):
        print(f"    {rank}. '{idx2word.get(i, UNK_TOKEN)}'  (p={p:.4f})")

    return idx2word.get(topk_idxs[0].item(), UNK_TOKEN)

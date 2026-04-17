import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class NextWordDataset(Dataset):
    """
    Converts sentences into multiple samples based on window size.
    Example: sentence [A, B, C, D] with window=2
    Sample 1: input=[PAD, A], target=B
    Sample 2: input=[A, B], target=C
    Sample 3: input=[B, C], target=D
    """
    def __init__(self, indexed_sentences: List[List[int]], pad_idx: int, window_size: int):
        self.pad_idx = pad_idx
        self.window_size = window_size
        self.samples = []
        
        for sentence in indexed_sentences:
            if len(sentence) < 2:
                continue
            for i in range(1, len(sentence)):
                target = sentence[i]
                context = sentence[max(0, i - window_size):i]
                # Pad if context is shorter than window_size
                if len(context) < window_size:
                    padded_context = [pad_idx] * (window_size - len(context)) + context
                else:
                    padded_context = context
                self.samples.append((padded_context, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp, tgt = self.samples[idx]
        return (torch.tensor(inp, dtype=torch.long),
                torch.tensor(tgt, dtype=torch.long))

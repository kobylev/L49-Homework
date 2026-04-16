import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class NextWordDataset(Dataset):
    """
    Converts each sentence  [w0 … wN]  into one sample:
        input  = [w0 … w_{N-1}]  padded to max_seq_len
        target = w_N             (scalar)
    """

    def __init__(self, indexed: List[List[int]], pad_idx: int, max_seq_len: int):
        self.pad_idx     = pad_idx
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[List[int], int]] = [
            (s[:-1], s[-1]) for s in indexed if len(s) >= 2
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp, tgt = self.samples[idx]
        seq_len  = min(len(inp), self.max_seq_len)
        padded   = [self.pad_idx] * self.max_seq_len
        padded[:seq_len] = inp[:seq_len]
        return (torch.tensor(padded, dtype=torch.long),
                torch.tensor(tgt,   dtype=torch.long))

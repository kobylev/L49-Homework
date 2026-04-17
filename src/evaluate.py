import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model import NextWordModel
import math
from typing import Tuple, Dict

@torch.no_grad()
def evaluate(model: NextWordModel,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float, float, Dict[int, float]]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    
    acc_by_len = {}
    count_by_len = {}
    
    for inputs, targets, s_lens in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits, _ = model(inputs)
        
        loss = criterion(logits, targets)
        total_loss += loss.item() * inputs.size(0)
        
        preds = logits.argmax(dim=-1)
        corrects = (preds == targets)
        total_correct += corrects.sum().item()
        total_n += inputs.size(0)
        
        for i in range(inputs.size(0)):
            sl = s_lens[i].item()
            if sl not in acc_by_len:
                acc_by_len[sl] = 0
                count_by_len[sl] = 0
            acc_by_len[sl] += corrects[i].item()
            count_by_len[sl] += 1
            
    avg_loss = total_loss / total_n
    avg_acc = total_correct / total_n
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    final_acc_by_len = {k: acc_by_len[k] / count_by_len[k] for k in acc_by_len}
    
    return avg_loss, avg_acc, perplexity, final_acc_by_len

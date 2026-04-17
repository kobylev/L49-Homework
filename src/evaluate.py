import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model import NextWordModel
import math
from typing import Tuple

@torch.no_grad()
def evaluate(model: NextWordModel,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits, _ = model(inputs)
        
        loss = criterion(logits, targets)
        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()
        total_n += inputs.size(0)
        
    avg_loss = total_loss / total_n
    avg_acc = total_correct / total_n
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    return avg_loss, avg_acc, perplexity

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model import NextWordModel
import time
from typing import List, Optional, Dict

def train_one_epoch(model: NextWordModel,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    grad_clip: float,
                    layer_grad_norms: Optional[Dict[str, List[float]]] = None) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    
    for inputs, targets, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        
        if layer_grad_norms is not None:
            # Track gradients for: embedding, rnn, fc
            for name, param in model.named_parameters():
                if param.grad is not None:
                    layer_key = "other"
                    if "embedding" in name: layer_key = "embedding"
                    elif "rnn" in name: layer_key = "rnn"
                    elif "fc" in name: layer_key = "fc"
                    
                    norm = param.grad.detach().data.norm(2).item()
                    if layer_key not in layer_grad_norms:
                        layer_grad_norms[layer_key] = []
                    layer_grad_norms[layer_key].append(norm)
            
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        total_n += inputs.size(0)
        
    return total_loss / total_n

class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
from models.next_word_model import NextWordModel
from training.metrics import ExperimentResult
from utils.config import print_memory_usage

def train_one_epoch(model:           NextWordModel,
                    loader:          DataLoader,
                    optim:           torch.optim.Optimizer,
                    criterion:       nn.Module,
                    device:          torch.device,
                    epoch:           int,
                    grad_clip:       float,
                    log_every:       int,
                    step_grad_norms: List[float] = None) -> float:
    """One full training pass.  Returns mean cross-entropy loss."""
    model.train()
    total_loss, total_n = 0.0, 0
    t0 = time.time()

    for step, (inputs, targets) in enumerate(loader):
        inputs  = inputs.to(device)
        targets = targets.to(device)

        optim.zero_grad()
        logits, _ = model(inputs)
        loss      = criterion(logits, targets)
        loss.backward()
        
        # Track gradient norm EXACTLY before clipping
        if step_grad_norms is not None:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            step_grad_norms.append(total_norm)

        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        total_loss += loss.item() * inputs.size(0)
        total_n    += inputs.size(0)

        if (step + 1) % log_every == 0:
            print(f"  [{model.rnn_type}] Epoch {epoch} | "
                  f"step {step+1:>5}/{len(loader)} | "
                  f"loss {total_loss/total_n:.4f} | "
                  f"elapsed {time.time()-t0:.1f}s")

    return total_loss / total_n

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

@torch.no_grad()
def evaluate(model:     NextWordModel,
             loader:    DataLoader,
             criterion: nn.Module,
             device:    torch.device) -> Tuple[float, float, float]:
    """Validation pass.  Returns (mean_loss, top-1_accuracy, perplexity)."""
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        logits, _ = model(inputs)
        
        loss = criterion(logits, targets)
        total_loss    += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()
        total_n       += inputs.size(0)

    avg_loss = total_loss / total_n
    avg_acc  = total_correct / total_n
    perplexity = math.exp(avg_loss)

    return avg_loss, avg_acc, perplexity

def run_experiment(rnn_type:     str,
                   train_loader: DataLoader,
                   val_loader:   DataLoader,
                   vocab_size:   int,
                   pad_idx:      int,
                   device:       torch.device,
                   cfg:          Dict) -> Tuple[ExperimentResult, NextWordModel]:
    """
    Instantiate, train, and validate one model variant.
    Returns a fully populated ExperimentResult and the best model.
    """
    print(f"\n{'═'*60}")
    print(f"  EXPERIMENT : {rnn_type}")
    print(f"{'═'*60}")

    model = NextWordModel(
        vocab_size    = vocab_size,
        embedding_dim = cfg["embedding_dim"],
        hidden_dim    = cfg["hidden_dim"],
        num_layers    = cfg["num_rnn_layers"],
        dropout       = cfg["dropout"],
        pad_idx       = pad_idx,
        rnn_type      = rnn_type,
    ).to(device)

    print(model)
    print(f"\n[{rnn_type}] Trainable parameters: {model.count_params():,}\n")
    print_memory_usage(f"{rnn_type} model init")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=cfg.get("patience", 3))

    result = ExperimentResult(
        rnn_type     = rnn_type,
        num_params   = model.count_params(),
        train_time_s = 0.0,
    )
    best_val_loss = float("inf")
    t_start       = time.time()
    cumulative_time = 0.0

    for epoch in range(1, cfg["num_epochs"] + 1):
        ep_start = time.time()
        print(f"\n{'─'*50}")
        print(f"  [{rnn_type}] Epoch {epoch}/{cfg['num_epochs']}")
        print(f"{'─'*50}")

        train_loss = train_one_epoch(
            model, train_loader, optimiser, criterion,
            device, epoch, cfg["grad_clip"], cfg["log_every_n_batches"],
            step_grad_norms=result.step_grad_norms)

        val_loss, val_acc, val_perp = evaluate(model, val_loader, criterion, device)
        ep_time = time.time() - ep_start
        cumulative_time += ep_time

        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)
        result.val_accs.append(val_acc)
        result.val_perplexities.append(val_perp)
        result.wall_clock_times.append(cumulative_time)

        print(f"\n  ► [{rnn_type}] Epoch {epoch} | "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_perp={val_perp:.2f}  "
              f"time={ep_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_{rnn_type.lower()}_model.pt")
            print(f"  ✓ New best {rnn_type} model saved.")

        # Check Early Stopping
        if early_stopping(val_loss):
            print(f"\n  [Early Stopping] No improvement in validation loss for {early_stopping.patience} epochs. Stopping...")
            break

        print_memory_usage(f"{rnn_type} end-of-epoch {epoch}")

    result.train_time_s = time.time() - t_start
    return result, model


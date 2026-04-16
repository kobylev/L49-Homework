"""
=============================================================================
RNN vs LSTM  —  Next-Word Prediction  —  PyTorch Implementation
=============================================================================
Assignment Coverage:
  1.  Synthetic dataset generation (10,000-word vocab, 100,000 sentences)
  2.  Tokenisation & indexing (word2idx / idx2word)
  3.  Dataset preparation — input/target pairs, 80/20 train-val split
  4.  Model — Embedding → {RNN | LSTM} → Linear (+ CrossEntropyLoss)
  5.  Training loop with Adam optimiser, validation evaluation
  6.  Prediction / decoding function
  7.  [NEW] Side-by-side experiment runner — trains both architectures
      on identical data and hyper-parameters, then produces:
        • Per-epoch loss / accuracy comparison table
        • Saved loss-curve PNG  (loss_curves.png)
        • Saved metrics CSV     (results.csv)
        • Statistical analysis  printed to stdout

All major hyper-parameters live in CONFIG so they can be tuned without
touching the rest of the code.
=============================================================================
"""

import csv
import random
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG: Dict = {
    # ── data ──────────────────────────────────────────────────────────────────
    "vocab_size":       10_000,   # unique words (excl. special tokens)
    "num_sentences":   100_000,   # total synthetic sentences
    "min_sentence_len":     5,    # shortest sentence in words
    "max_sentence_len":     7,    # longest  sentence in words
    "train_split":       0.80,    # fraction used for training

    # ── model (shared between RNN & LSTM) ─────────────────────────────────────
    "embedding_dim":      128,    # word-embedding size
    "hidden_dim":         256,    # recurrent hidden-state width
    "num_rnn_layers":       2,    # stacked recurrent layers
    "dropout":           0.30,    # inter-layer dropout (only if layers > 1)

    # ── training ──────────────────────────────────────────────────────────────
    "num_epochs":           5,
    "batch_size":         512,
    "learning_rate":    1e-3,
    "grad_clip":         5.0,     # max gradient norm

    # ── misc ──────────────────────────────────────────────────────────────────
    "seed":                42,
    "log_every_n_batches": 200,
    "loss_curve_path":    "loss_curves.png",
    "results_csv_path":   "results.csv",
}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_vocabulary(vocab_size: int, seed: int) -> List[str]:
    """
    Build `vocab_size` unique pseudo-English words via combinatorial
    phoneme concatenation.  Sorted output guarantees reproducibility.
    """
    consonants = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m",
                  "n", "p", "r", "s", "t", "v", "w", "z",
                  "br", "cr", "dr", "fl", "gr", "pr", "st", "tr"]
    vowels     = ["a", "e", "i", "o", "u", "ae", "ou", "ai", "ea"]
    endings    = ["", "s", "ed", "ing", "er", "ly", "tion", "al", "ous", "ment"]

    vocab: set = set()
    rng = random.Random(seed)
    while len(vocab) < vocab_size:
        word = (rng.choice(consonants) + rng.choice(vowels)
                + rng.choice(consonants) + rng.choice(vowels)
                + rng.choice(endings))
        vocab.add(word)

    vocab_list = sorted(vocab)
    print(f"[Data]  Vocabulary: {len(vocab_list):,} unique words generated.")
    return vocab_list


def build_vocab_maps(vocab_words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Index 0 → <PAD>  (embedding kept at zero, ignored by loss)
    Index 1 → <UNK>
    Index 2+ → real words
    """
    word2idx: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for w in vocab_words:
        word2idx[w] = len(word2idx)
    idx2word = {v: k for k, v in word2idx.items()}
    print(f"[Data]  Vocab maps ready. Total tokens (incl. specials): {len(word2idx):,}")
    return word2idx, idx2word


def generate_sentences(vocab_words: List[str],
                        num_sentences: int,
                        min_len: int,
                        max_len: int,
                        seed: int) -> List[List[str]]:
    """Randomly assemble sentences from the vocabulary."""
    rng = random.Random(seed + 1)
    sentences = [rng.choices(vocab_words, k=rng.randint(min_len, max_len))
                 for _ in range(num_sentences)]
    avg = sum(len(s) for s in sentences) / len(sentences)
    print(f"[Data]  {num_sentences:,} sentences generated. Avg length: {avg:.2f} words.")
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def tokenise(sentences: List[List[str]],
             word2idx: Dict[str, int]) -> List[List[int]]:
    """Word strings → integer index sequences (unknown words → UNK)."""
    unk = word2idx[UNK_TOKEN]
    return [[word2idx.get(w, unk) for w in s] for s in sentences]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODEL — supports both nn.RNN and nn.LSTM via rnn_type argument
# ─────────────────────────────────────────────────────────────────────────────

class NextWordModel(nn.Module):
    """
    Shared architecture for Vanilla-RNN and LSTM.

    Embedding  ──►  {RNN | LSTM}  ──►  last time-step  ──►  Linear  ──►  logits

    The only difference between the two modes is:
      • nn.RNN  returns  (output, h_n)           — hidden state is a tensor
      • nn.LSTM returns  (output, (h_n, c_n))    — hidden state is a tuple

    Both paths are handled transparently inside `forward`.

    Parameters
    ----------
    vocab_size    : total token count (incl. PAD & UNK)
    embedding_dim : word-embedding width
    hidden_dim    : recurrent hidden-state width
    num_layers    : stacked recurrent layers
    dropout       : inter-layer dropout
    pad_idx       : PAD token index (embedding frozen to 0)
    rnn_type      : "RNN" or "LSTM"
    """

    def __init__(self,
                 vocab_size:    int,
                 embedding_dim: int,
                 hidden_dim:    int,
                 num_layers:    int,
                 dropout:       float,
                 pad_idx:       int,
                 rnn_type:      str = "RNN"):
        super().__init__()
        assert rnn_type in ("RNN", "LSTM"), "rnn_type must be 'RNN' or 'LSTM'"

        self.rnn_type   = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ── Embedding ─────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=pad_idx)

        # ── Recurrent core ────────────────────────────────────────────────────
        rnn_drop = dropout if num_layers > 1 else 0.0
        shared_kwargs = dict(
            input_size  = embedding_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = rnn_drop,
        )
        if rnn_type == "RNN":
            self.rnn = nn.RNN(**shared_kwargs, nonlinearity="tanh")
        else:
            self.rnn = nn.LSTM(**shared_kwargs)

        # ── Output projection ─────────────────────────────────────────────────
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # LSTM forget-gate bias trick: initialise to 1 to help
                # remember long-range context at training start
                if self.rnn_type == "LSTM":
                    hidden = param.data.size(0) // 4
                    param.data[hidden:2 * hidden].fill_(1.0)

        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Union[torch.Tensor,
                               Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor,
               Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        x      : (batch, seq_len)         integer token indices
        hidden : previous hidden state or None (initialised to zeros)

        Returns
        -------
        logits : (batch, vocab_size)       raw un-normalised scores
        hidden : updated hidden state (detached — no cross-batch BPTT)
        """
        emb = self.embedding(x)                  # (batch, seq, emb_dim)
        out, hidden = self.rnn(emb, hidden)       # out: (batch, seq, hidden)
        last = out[:, -1, :]                      # (batch, hidden)
        logits = self.fc(last)                    # (batch, vocab_size)

        # Detach to prevent gradients flowing across batch boundaries
        if self.rnn_type == "LSTM":
            hidden = (hidden[0].detach(), hidden[1].detach())
        else:
            hidden = hidden.detach()

        return logits, hidden

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model:     NextWordModel,
                    loader:    DataLoader,
                    optim:     torch.optim.Optimizer,
                    criterion: nn.Module,
                    device:    torch.device,
                    epoch:     int,
                    grad_clip: float,
                    log_every: int) -> float:
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


@torch.no_grad()
def evaluate(model:     NextWordModel,
             loader:    DataLoader,
             criterion: nn.Module,
             device:    torch.device) -> Tuple[float, float]:
    """Validation pass.  Returns (mean_loss, top-1_accuracy)."""
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        logits, _ = model(inputs)
        total_loss    += criterion(logits, targets).item() * inputs.size(0)
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()
        total_n       += inputs.size(0)

    return total_loss / total_n, total_correct / total_n


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PREDICTION & DECODING
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 7.  RESULTS ANALYSIS  (new section)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """Container for one model's full training run metrics."""
    rnn_type:         str
    num_params:       int
    train_time_s:     float
    train_losses:     List[float] = field(default_factory=list)
    val_losses:       List[float] = field(default_factory=list)
    val_accs:         List[float] = field(default_factory=list)

    # ── derived properties ────────────────────────────────────────────────────
    @property
    def best_val_loss(self) -> float:
        return min(self.val_losses) if self.val_losses else float("inf")

    @property
    def best_val_acc(self) -> float:
        return max(self.val_accs) if self.val_accs else 0.0

    @property
    def final_train_loss(self) -> float:
        return self.train_losses[-1] if self.train_losses else float("inf")

    @property
    def final_val_loss(self) -> float:
        return self.val_losses[-1] if self.val_losses else float("inf")

    @property
    def loss_improvement(self) -> float:
        """Total absolute drop in training loss from epoch 1 → last."""
        if len(self.train_losses) < 2:
            return 0.0
        return self.train_losses[0] - self.train_losses[-1]

    @property
    def overfitting_gap(self) -> float:
        """Final (val_loss − train_loss).  Positive = generalisation gap."""
        return self.final_val_loss - self.final_train_loss


def print_comparison_table(results: List[ExperimentResult]) -> None:
    """Pretty-print a side-by-side metric table to stdout."""
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  MODEL COMPARISON — RNN vs LSTM")
    print(sep)
    metrics = [
        ("Architecture",          lambda r: r.rnn_type),
        ("Trainable Parameters",  lambda r: f"{r.num_params:,}"),
        ("Training Time (s)",     lambda r: f"{r.train_time_s:.1f}"),
        ("Train Loss (final)",    lambda r: f"{r.final_train_loss:.4f}"),
        ("Val   Loss (final)",    lambda r: f"{r.final_val_loss:.4f}"),
        ("Best  Val Loss",        lambda r: f"{r.best_val_loss:.4f}"),
        ("Best  Val Accuracy",    lambda r: f"{r.best_val_acc*100:.2f}%"),
        ("Loss Improvement Δ",    lambda r: f"{r.loss_improvement:.4f}"),
        ("Overfitting Gap",       lambda r: f"{r.overfitting_gap:+.4f}"),
    ]
    col_w = 26
    print(f"  {'Metric':<30}", end="")
    for r in results:
        print(f"{r.rnn_type:>{col_w}}", end="")
    print()
    print(f"  {'─'*30}", end="")
    for _ in results:
        print("─" * col_w, end="")
    print()
    for label, fn in metrics:
        print(f"  {label:<30}", end="")
        for r in results:
            print(f"{fn(r):>{col_w}}", end="")
        print()
    print(sep)

    # ── Per-epoch breakdown ───────────────────────────────────────────────────
    epochs = range(1, len(results[0].train_losses) + 1)
    print(f"\n  Per-Epoch Training Loss")
    header = f"  {'Epoch':>6}  " + "  ".join(
        f"{r.rnn_type + ' Train':>14}  {r.rnn_type + ' Val':>14}"
        for r in results)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for ep in epochs:
        row = f"  {ep:>6}  "
        for r in results:
            row += (f"  {r.train_losses[ep-1]:>14.4f}"
                    f"  {r.val_losses[ep-1]:>14.4f}")
        print(row)
    print(sep + "\n")

    # ── Narrative analysis ────────────────────────────────────────────────────
    rnn_r, lstm_r = results[0], results[1]
    winner = "LSTM" if lstm_r.best_val_loss < rnn_r.best_val_loss else "RNN"
    delta_loss = abs(rnn_r.best_val_loss - lstm_r.best_val_loss)
    delta_acc  = abs(rnn_r.best_val_acc  - lstm_r.best_val_acc) * 100
    speedup    = rnn_r.train_time_s / lstm_r.train_time_s

    print("  ── ANALYSIS ──")
    print(f"  • Best validation loss:  RNN={rnn_r.best_val_loss:.4f}  "
          f"LSTM={lstm_r.best_val_loss:.4f}  → {winner} wins by {delta_loss:.4f}")
    print(f"  • Best validation acc :  RNN={rnn_r.best_val_acc*100:.2f}%  "
          f"LSTM={lstm_r.best_val_acc*100:.2f}%  → Δ={delta_acc:.2f}pp")
    print(f"  • Parameter count     :  RNN={rnn_r.num_params:,}  "
          f"LSTM={lstm_r.num_params:,}  "
          f"(LSTM has {(lstm_r.num_params/rnn_r.num_params - 1)*100:.0f}% more params)")
    print(f"  • Training speed      :  RNN={rnn_r.train_time_s:.1f}s  "
          f"LSTM={lstm_r.train_time_s:.1f}s  "
          f"(RNN is {speedup:.2f}× faster)" if speedup >= 1
          else f"  • Training speed      :  LSTM is {1/speedup:.2f}× faster)")
    print(f"  • Overfitting gap     :  RNN={rnn_r.overfitting_gap:+.4f}  "
          f"LSTM={lstm_r.overfitting_gap:+.4f}")
    print(f"  • Loss Δ (ep1→last)   :  RNN={rnn_r.loss_improvement:.4f}  "
          f"LSTM={lstm_r.loss_improvement:.4f}")
    print()


def save_results_csv(results: List[ExperimentResult], path: str) -> None:
    """Write per-epoch metrics for both models to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epoch",
                         "train_loss", "val_loss", "val_acc_pct"])
        for r in results:
            for ep, (tl, vl, va) in enumerate(
                    zip(r.train_losses, r.val_losses, r.val_accs), 1):
                writer.writerow([r.rnn_type, ep,
                                 f"{tl:.6f}", f"{vl:.6f}",
                                 f"{va*100:.4f}"])
    print(f"[Results] Metrics saved → {path}")


def plot_loss_curves(results: List[ExperimentResult], path: str) -> None:
    """
    Produce a 2×2 subplot figure:
      Top-left   : Training loss curves for both models
      Top-right  : Validation loss curves for both models
      Bottom-left: Validation accuracy curves
      Bottom-right: Overfitting gap (val − train) per epoch
    """
    epochs  = list(range(1, len(results[0].train_losses) + 1))
    colours = {"RNN": "#E07B54", "LSTM": "#4C72B0"}
    styles  = {"RNN": "o-",      "LSTM": "s--"}

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("RNN vs LSTM — Next-Word Prediction\nLoss & Accuracy Comparison",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.38, wspace=0.32)

    ax_tl = fig.add_subplot(gs[0, 0])   # training loss
    ax_tr = fig.add_subplot(gs[0, 1])   # validation loss
    ax_bl = fig.add_subplot(gs[1, 0])   # validation accuracy
    ax_br = fig.add_subplot(gs[1, 1])   # overfitting gap

    for r in results:
        c, s = colours[r.rnn_type], styles[r.rnn_type]
        gap  = [vl - tl for vl, tl in zip(r.val_losses, r.train_losses)]

        ax_tl.plot(epochs, r.train_losses, s, color=c,
                   label=r.rnn_type, linewidth=2, markersize=7)
        ax_tr.plot(epochs, r.val_losses,   s, color=c,
                   label=r.rnn_type, linewidth=2, markersize=7)
        ax_bl.plot(epochs, [a * 100 for a in r.val_accs], s, color=c,
                   label=r.rnn_type, linewidth=2, markersize=7)
        ax_br.plot(epochs, gap, s, color=c,
                   label=r.rnn_type, linewidth=2, markersize=7)

    for ax, title, ylabel in [
        (ax_tl, "Training Loss    (CrossEntropy)",   "Loss"),
        (ax_tr, "Validation Loss  (CrossEntropy)",   "Loss"),
        (ax_bl, "Validation Top-1 Accuracy",          "Accuracy (%)"),
        (ax_br, "Overfitting Gap  (Val − Train)",     "Gap"),
    ]:
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(epochs)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_br.axhline(0, color="grey", linewidth=1, linestyle=":")  # zero-gap line

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Results] Loss-curve plot saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_memory_usage(label: str) -> None:
    _, peak = tracemalloc.get_traced_memory()
    print(f"[Memory] {label}: peak RAM = {peak / 1024**2:.1f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  EXPERIMENT RUNNER  (train one model, return ExperimentResult)
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(rnn_type:     str,
                   train_loader: DataLoader,
                   val_loader:   DataLoader,
                   vocab_size:   int,
                   pad_idx:      int,
                   device:       torch.device,
                   cfg:          Dict) -> ExperimentResult:
    """
    Instantiate, train, and validate one model variant.
    Returns a fully populated ExperimentResult.
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

    result = ExperimentResult(
        rnn_type     = rnn_type,
        num_params   = model.count_params(),
        train_time_s = 0.0,
    )
    best_val_loss = float("inf")
    t_start       = time.time()

    for epoch in range(1, cfg["num_epochs"] + 1):
        ep_start = time.time()
        print(f"\n{'─'*50}")
        print(f"  [{rnn_type}] Epoch {epoch}/{cfg['num_epochs']}")
        print(f"{'─'*50}")

        train_loss = train_one_epoch(
            model, train_loader, optimiser, criterion,
            device, epoch, cfg["grad_clip"], cfg["log_every_n_batches"])

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        ep_time = time.time() - ep_start

        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)
        result.val_accs.append(val_acc)

        print(f"\n  ► [{rnn_type}] Epoch {epoch} | "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc*100:.2f}%  "
              f"time={ep_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       f"best_{rnn_type.lower()}_model.pt")
            print(f"  ✓ New best {rnn_type} model saved.")

        print_memory_usage(f"{rnn_type} end-of-epoch {epoch}")

    result.train_time_s = time.time() - t_start
    return result, model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg  = CONFIG
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  RNN vs LSTM — Next-Word Prediction")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    tracemalloc.start()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("── Step 1 : Data Generation ──")
    vocab_words        = generate_vocabulary(cfg["vocab_size"], seed)
    word2idx, idx2word = build_vocab_maps(vocab_words)
    actual_vocab_size  = len(word2idx)
    print(f"[Data]  Actual vocab (with specials): {actual_vocab_size:,}\n")

    sentences = generate_sentences(
        vocab_words, cfg["num_sentences"],
        cfg["min_sentence_len"], cfg["max_sentence_len"], seed)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n── Step 2 : Tokenisation ──")
    indexed = tokenise(sentences, word2idx)
    print(f"[Prep]  Example:  {sentences[0]}")
    print(f"[Prep]  Indexed:  {indexed[0]}\n")
    print_memory_usage("After tokenisation")

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    print("\n── Step 3 : Dataset Preparation ──")
    max_seq_len = cfg["max_sentence_len"] - 1
    pad_idx     = word2idx[PAD_TOKEN]

    full_ds    = NextWordDataset(indexed, pad_idx, max_seq_len)
    total      = len(full_ds)
    train_size = int(total * cfg["train_split"])
    val_size   = total - train_size

    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed))

    print(f"[Data]  Total={total:,}  Train={train_size:,}  Val={val_size:,}\n")

    # Shared loaders — both models see exactly the same data
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=0,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=0,
                              pin_memory=(device.type == "cuda"))

    # ── 4–6. Train both models ────────────────────────────────────────────────
    rnn_result,  rnn_model  = run_experiment(
        "RNN",  train_loader, val_loader,
        actual_vocab_size, pad_idx, device, cfg)

    lstm_result, lstm_model = run_experiment(
        "LSTM", train_loader, val_loader,
        actual_vocab_size, pad_idx, device, cfg)

    results = [rnn_result, lstm_result]

    # ── 7. Analysis & Outputs ─────────────────────────────────────────────────
    print("\n── Step 7 : Results Analysis ──")
    print_comparison_table(results)
    save_results_csv(results, cfg["results_csv_path"])
    plot_loss_curves(results,  cfg["loss_curve_path"])

    # ── 8. Prediction demo ────────────────────────────────────────────────────
    print("\n── Step 8 : Prediction Demo ──")
    rng = random.Random(seed + 99)
    sample_sentences = [sentences[i]
                        for i in rng.sample(range(len(sentences)), 3)]

    for sent in sample_sentences:
        seed_words  = sent[:-1]
        true_target = sent[-1]
        print(f"\n  Seed: {seed_words}  →  True next word: '{true_target}'")
        for model, result in [(rnn_model, rnn_result),
                               (lstm_model, lstm_result)]:
            model.load_state_dict(torch.load(
                f"best_{result.rnn_type.lower()}_model.pt",
                map_location=device, weights_only=True))
            pred  = predict_next_word(
                seed_words, model, word2idx, idx2word,
                device, max_seq_len, top_k=3)
            match = "✓" if pred == true_target else "✗"
            print(f"  [{result.rnn_type}] Predicted: '{pred}' {match}")

    tracemalloc.stop()
    print("\nAll done.  Outputs: loss_curves.png | results.csv | best_*.pt\n")


if __name__ == "__main__":
    main()

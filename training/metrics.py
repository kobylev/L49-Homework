import csv
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

@dataclass
class ExperimentResult:
    """Container for one model's full training run metrics."""
    rnn_type:         str
    num_params:       int
    train_time_s:     float
    train_losses:     List[float] = field(default_factory=list)
    val_losses:       List[float] = field(default_factory=list)
    val_accs:         List[float] = field(default_factory=list)
    step_grad_norms:  List[float] = field(default_factory=list)
    val_perplexities: List[float] = field(default_factory=list)
    wall_clock_times: List[float] = field(default_factory=list)

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
        if len(self.train_losses) < 2:
            return 0.0
        return self.train_losses[0] - self.train_losses[-1]

    @property
    def overfitting_gap(self) -> float:
        return self.final_val_loss - self.final_train_loss

def print_comparison_table(results: List[ExperimentResult]) -> None:
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

def save_results_csv(results: List[ExperimentResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epoch", "train_loss", "val_loss", "val_acc_pct"])
        for r in results:
            for ep, (tl, vl, va) in enumerate(zip(r.train_losses, r.val_losses, r.val_accs), 1):
                writer.writerow([r.rnn_type, ep, f"{tl:.6f}", f"{vl:.6f}", f"{va*100:.4f}"])
    print(f"[Results] Metrics saved -> {path}")

def plot_advanced_metrics(results: List[ExperimentResult], path: str) -> None:
    """Creates a 2x2 grid of advanced metrics for RNN vs LSTM."""
    epochs  = list(range(1, len(results[0].train_losses) + 1))
    colours = {"RNN": "#E07B54", "LSTM": "#4C72B0"}
    styles  = {"RNN": "o-",      "LSTM": "s--"}

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle("RNN vs LSTM — Advanced Training Metrics", 
                 fontsize=16, fontweight="bold", y=0.96)
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_grad = fig.add_subplot(gs[0, 1])
    ax_perp = fig.add_subplot(gs[1, 0])
    ax_time = fig.add_subplot(gs[1, 1])

    for r in results:
        c = colours[r.rnn_type]
        s = styles[r.rnn_type]
        
        # 1. Training & Validation Loss
        ax_loss.plot(epochs, r.train_losses, s, color=c, label=f"{r.rnn_type} Train", alpha=0.9)
        ax_loss.plot(epochs, r.val_losses, ":", color=c, label=f"{r.rnn_type} Val", alpha=0.7)
        
        # 2. Gradient Norms (Step-wise)
        steps = list(range(1, len(r.step_grad_norms) + 1))
        ax_grad.plot(steps, r.step_grad_norms, color=c, label=r.rnn_type, alpha=0.6, linewidth=0.8)
        
        # 3. Validation Perplexity
        ax_perp.plot(epochs, r.val_perplexities, s, color=c, label=r.rnn_type)
        
        # 4. Wall-clock Time vs Loss
        ax_time.plot(r.wall_clock_times, r.val_losses, s, color=c, label=r.rnn_type)

    # Styling
    ax_loss.set_title("1. Training & Validation Loss", fontsize=12, pad=10)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    ax_grad.set_title("2. Gradient Norm Stability (pre-clipping)", fontsize=12, pad=10)
    ax_grad.set_xlabel("Training Step")
    ax_grad.set_ylabel("L2 Norm")
    ax_grad.axhline(5.0, color="red", linestyle="--", alpha=0.5, label="Clip Threshold (5.0)")
    ax_grad.legend(fontsize=9)
    ax_grad.grid(True, alpha=0.3)

    ax_perp.set_title("3. Validation Perplexity", fontsize=12, pad=10)
    ax_perp.set_xlabel("Epoch")
    ax_perp.set_ylabel("Perplexity")
    ax_perp.legend(fontsize=9)
    ax_perp.grid(True, alpha=0.3)

    ax_time.set_title("4. Validation Loss vs. Wall-clock Time", fontsize=12, pad=10)
    ax_time.set_xlabel("Cumulative Time (seconds)")
    ax_time.set_ylabel("Validation Loss")
    ax_time.legend(fontsize=9)
    ax_time.grid(True, alpha=0.3)

    for ax in [ax_loss, ax_grad, ax_perp, ax_time]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Results] Advanced metrics plot saved -> {path}")

def plot_loss_curves(results: List[ExperimentResult], path: str) -> None:
    epochs  = list(range(1, len(results[0].train_losses) + 1))
    colours = {"RNN": "#E07B54", "LSTM": "#4C72B0"}
    styles  = {"RNN": "o-",      "LSTM": "s--"}

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("RNN vs LSTM — Next-Word Prediction\nLoss & Accuracy Comparison",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    for r in results:
        c, s = colours[r.rnn_type], styles[r.rnn_type]
        gap  = [vl - tl for vl, tl in zip(r.val_losses, r.train_losses)]

        ax_tl.plot(epochs, r.train_losses, s, color=c, label=r.rnn_type, linewidth=2, markersize=7)
        ax_tr.plot(epochs, r.val_losses,   s, color=c, label=r.rnn_type, linewidth=2, markersize=7)
        ax_bl.plot(epochs, [a * 100 for a in r.val_accs], s, color=c, label=r.rnn_type, linewidth=2, markersize=7)
        ax_br.plot(epochs, gap, s, color=c, label=r.rnn_type, linewidth=2, markersize=7)

    for ax, title, ylabel in [
        (ax_tl, "Training Loss (CrossEntropy)",   "Loss"),
        (ax_tr, "Validation Loss (CrossEntropy)",   "Loss"),
        (ax_bl, "Validation Top-1 Accuracy",          "Accuracy (%)"),
        (ax_br, "Overfitting Gap (Val - Train)",     "Gap"),
    ]:
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(epochs)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_br.axhline(0, color="grey", linewidth=1, linestyle=":")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Results] Loss-curve plot saved -> {path}")

import tracemalloc
from typing import Dict

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

CONFIG: Dict = {
    # ── data ──────────────────────────────────────────────────────────────────
    "vocab_size":       10_000,   # unique words (excl. special tokens)
    "num_sentences":    100_000,   # total synthetic sentences (increased to meet requirements)
    "min_sentence_len":     5,    # shortest sentence in words
    "max_sentence_len":    20,    # longest  sentence in words (increased as requested)
    "train_split":       0.80,    # fraction used for training

    # ── model (shared between RNN & LSTM) ─────────────────────────────────────
    "embedding_dim":      128,    # word-embedding size
    "hidden_dim":         256,    # recurrent hidden-state width
    "num_rnn_layers":       2,    # stacked recurrent layers
    "dropout":           0.30,    # inter-layer dropout (only if layers > 1)

    # ── training ──────────────────────────────────────────────────────────────
    "num_epochs":          10,
    "batch_size":         512,
    "learning_rate":    1e-3,
    "grad_clip":         5.0,     # max gradient norm
    "patience":            3,     # Early Stopping patience

    # ── misc ──────────────────────────────────────────────────────────────────
    "seed":                42,
    "log_every_n_batches": 200,
    "loss_curve_path":    "loss_curves.png",
    "advanced_metrics_path": "advanced_metrics.png",
    "results_csv_path":   "results.csv",
}

def print_memory_usage(label: str) -> None:
    """Utility to print current peak memory usage."""
    _, peak = tracemalloc.get_traced_memory()
    print(f"[Memory] {label}: peak RAM = {peak / 1024**2:.1f} MB")

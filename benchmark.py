import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os

# Import existing project modules
from utils.config import CONFIG, PAD_TOKEN
from utils.helpers import set_seed
from data.processor import generate_vocabulary, build_vocab_maps, generate_sentences, tokenise
from data.dataset import NextWordDataset
from training.trainer import run_experiment

def benchmark_regime(name, min_len, max_len, device, cfg):
    print(f"\n{'='*60}")
    print(f"  BENCHMARK REGIME: {name} (Length {min_len}-{max_len})")
    print(f"{'='*60}")
    
    # 1. Data Generation for this specific length
    vocab_words = generate_vocabulary(cfg["vocab_size"], cfg["seed"])
    word2idx, idx2word = build_vocab_maps(vocab_words)
    actual_vocab_size = len(word2idx)
    pad_idx = word2idx[PAD_TOKEN]
    
    sentences = generate_sentences(
        vocab_words, 20000, # Increased for more robust results
        min_len, max_len, cfg["seed"])

    indexed = tokenise(sentences, word2idx)
    max_seq_len = max_len - 1
    full_ds = NextWordDataset(indexed, pad_idx, max_seq_len)
    
    train_size = int(len(full_ds) * 0.8)
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

    results = {}
    for arch in ["RNN", "LSTM"]:
        # Run a mini-experiment (2 epochs for speed)
        temp_cfg = cfg.copy()
        temp_cfg["num_epochs"] = 2
        res, _ = run_experiment(arch, train_loader, val_loader, actual_vocab_size, pad_idx, device, temp_cfg)
        results[arch] = res
        
    return results

def main():
    cfg = CONFIG.copy()
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Compare Short vs Long
    short_results = benchmark_regime("Short", 5, 7, device, cfg)
    long_results = benchmark_regime("Long", 19, 20, device, cfg)
    
    # 2. Extract Data for Plotting
    labels = ['RNN (Short)', 'LSTM (Short)', 'RNN (Long)', 'LSTM (Long)']
    val_perps = [
        short_results["RNN"].val_perplexities[-1],
        short_results["LSTM"].val_perplexities[-1],
        long_results["RNN"].val_perplexities[-1],
        long_results["LSTM"].val_perplexities[-1]
    ]
    train_times = [
        short_results["RNN"].train_time_s,
        short_results["LSTM"].train_time_s,
        long_results["RNN"].train_time_s,
        long_results["LSTM"].train_time_s
    ]

    # 3. Create Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Perplexity
    colors = ['#E07B54', '#4C72B0', '#E07B54', '#4C72B0']
    bars1 = ax1.bar(labels, val_perps, color=colors, alpha=0.8)
    ax1.set_title('Validation Perplexity (Lower is Better)')
    ax1.set_ylabel('Perplexity')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot Training Time
    bars2 = ax2.bar(labels, train_times, color=colors, alpha=0.8)
    ax2.set_title('Training Time (Total for 2 Epochs)')
    ax2.set_ylabel('Time (seconds)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('comparison_results.png')
    print("\n[Benchmark] Results saved to comparison_results.png")
    
    # 4. Final Proof Summary
    print("\n" + "="*60)
    print("  VANISHING GRADIENT PROOF SUMMARY")
    print("="*60)
    print(f"RNN Short Perplexity: {short_results['RNN'].val_perplexities[-1]:.2f}")
    print(f"RNN Long Perplexity:  {long_results['RNN'].val_perplexities[-1]:.2f}")
    print(f"LSTM Long Perplexity: {long_results['LSTM'].val_perplexities[-1]:.2f}")
    
    diff_rnn = long_results["RNN"].val_perplexities[-1] - short_results["RNN"].val_perplexities[-1]
    diff_lstm = long_results["LSTM"].val_perplexities[-1] - short_results["LSTM"].val_perplexities[-1]
    
    print(f"\nRNN Degradation on Long Sequences: +{diff_rnn:.2f} PP")
    print(f"LSTM Degradation on Long Sequences: +{diff_lstm:.2f} PP")
    
    if diff_rnn > diff_lstm:
        print("\nCONCLUSION: LSTM is significantly more stable on long sequences.")
    print("="*60)

if __name__ == "__main__":
    main()

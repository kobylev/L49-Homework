"""
=============================================================================
RNN vs LSTM — Next-Word Prediction — Orchestrator
=============================================================================
Refactored into a modular architecture for scalability and clarity.
Modules:
  - utils.config: Configuration and constants
  - data.processor: Synthetic data generation and tokenisation
  - data.dataset: PyTorch Dataset
  - models.next_word_model: Architecture definitions
  - training.trainer: Training loops and experiment runner
  - training.metrics: Result tracking, CSV saving, and plotting
  - utils.helpers: Seeding and prediction demos
=============================================================================
"""

import random
import tracemalloc
import torch
from torch.utils.data import DataLoader, random_split

# Import modules
from utils.config import CONFIG, PAD_TOKEN, print_memory_usage
from utils.helpers import set_seed, predict_next_word
from data.processor import generate_vocabulary, build_vocab_maps, generate_sentences, tokenise
from data.dataset import NextWordDataset
from training.trainer import run_experiment
from training.metrics import print_comparison_table, save_results_csv, plot_loss_curves, plot_advanced_metrics

def main() -> None:
    # 0. Setup
    cfg  = CONFIG
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"  RNN vs LSTM — Next-Word Prediction (Modular)")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    tracemalloc.start()

    # 1. Data Generation
    print("── Step 1 : Data Generation ──")
    vocab_words        = generate_vocabulary(cfg["vocab_size"], cfg["seed"])
    word2idx, idx2word = build_vocab_maps(vocab_words)
    actual_vocab_size  = len(word2idx)
    
    sentences = generate_sentences(
        vocab_words, cfg["num_sentences"],
        cfg["min_sentence_len"], cfg["max_sentence_len"], cfg["seed"])

    # 2. Tokenisation
    print("\n── Step 2 : Tokenisation ──")
    indexed = tokenise(sentences, word2idx)
    print_memory_usage("After tokenisation")

    # 3. Dataset Preparation ──
    print("\n── Step 3 : Dataset Preparation ──")
    max_seq_len = cfg["max_sentence_len"] - 1
    pad_idx     = word2idx[PAD_TOKEN]

    full_ds    = NextWordDataset(indexed, pad_idx, max_seq_len)
    train_size = int(len(full_ds) * cfg["train_split"])
    test_size   = len(full_ds) - train_size

    train_ds, test_ds = random_split(
        full_ds, [train_size, test_size],
        generator=torch.Generator().manual_seed(cfg["seed"]))

    print(f"[Data]  Total={len(full_ds):,}  Train={train_size:,}  Test={test_size:,}\n")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=0,
                              pin_memory=(device.type == "cuda"))
    test_loader   = DataLoader(test_ds, batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=0,
                              pin_memory=(device.type == "cuda"))

    # 4. Run Experiments
    rnn_result, rnn_model = run_experiment(
        "RNN", train_loader, test_loader, actual_vocab_size, pad_idx, device, cfg)

    lstm_result, lstm_model = run_experiment(
        "LSTM", train_loader, test_loader, actual_vocab_size, pad_idx, device, cfg)


    results = [rnn_result, lstm_result]

    # 5. Analysis & Outputs
    print("\n── Step 5 : Results Analysis ──")
    print_comparison_table(results)
    save_results_csv(results, cfg["results_csv_path"])
    plot_loss_curves(results, cfg["loss_curve_path"])
    plot_advanced_metrics(results, cfg["advanced_metrics_path"])

    # 6. Prediction Demo
    print("\n── Step 6 : Prediction Demo ──")
    rng = random.Random(cfg["seed"] + 99)
    sample_indices = rng.sample(range(len(sentences)), 3)
    
    for idx in sample_indices:
        sent = sentences[idx]
        seed_words  = sent[:-1]
        true_target = sent[-1]
        print(f"\n  Seed: {seed_words}  ->  True next word: '{true_target}'")
        
        for model, res in [(rnn_model, rnn_result), (lstm_model, lstm_result)]:
            # Load best weights
            model.load_state_dict(torch.load(f"best_{res.rnn_type.lower()}_model.pt", 
                                           map_location=device, weights_only=True))
            pred = predict_next_word(seed_words, model, word2idx, idx2word, 
                                     device, max_seq_len, top_k=3)
            match = "✓" if pred == true_target else "✗"
            print(f"  [{res.rnn_type}] Predicted: '{pred}' {match}")

    tracemalloc.stop()
    print("\nPipeline Complete. Outputs: loss_curves.png | results.csv | best_*.pt\n")

if __name__ == "__main__":
    main()

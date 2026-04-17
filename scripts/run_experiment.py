import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing import (generate_structured_vocabulary, 
                               generate_structured_sentences, 
                               build_vocab_maps, tokenize, PAD_TOKEN)
from src.dataset import NextWordDataset
from src.model import NextWordModel
from src.train import train_one_epoch, EarlyStopping
from src.evaluate import evaluate
import time
import argparse
import numpy as np

def run_single_experiment(rnn_type, window_size, dataset_type, config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning experiment: {rnn_type}, window={window_size}, dataset={dataset_type}")
    
    # Data generation
    vocab_pools = generate_structured_vocabulary(config['data']['vocab_size'], config['data']['seed'])
    
    if dataset_type == "short":
        num_sentences = config['data']['num_sentences']
        # The generator naturally produces sentences of length 3 or 4.
        # Pattern: sub + verb + obj [+ mod]
        sentences = generate_structured_sentences(vocab_pools, num_sentences, config['data']['seed'])
    else:
        # Long sentences: we can concatenate multiple patterns or just repeat
        # For simplicity, let's just generate a lot and join them to reach 15-20 words
        num_sentences = config['data']['num_sentences']
        raw_sentences = generate_structured_sentences(vocab_pools, num_sentences * 5, config['data']['seed'])
        sentences = []
        for i in range(num_sentences):
            long_s = []
            while len(long_s) < 15:
                long_s.extend(raw_sentences[i * 5 + len(long_s)//4])
            sentences.append(long_s[:20]) # Limit to 20

    word2idx, idx2word = build_vocab_maps(vocab_pools)
    indexed_sentences = tokenize(sentences, word2idx)
    
    # Split 80/10/10
    total = len(indexed_sentences)
    train_size = int(config['data']['train_ratio'] * total)
    val_size = int(config['data']['val_ratio'] * total)
    test_size = total - train_size - val_size
    
    train_indices, val_indices, test_indices = random_split(
        range(total), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config['data']['seed'])
    )
    
    train_sentences = [indexed_sentences[i] for i in train_indices]
    val_sentences = [indexed_sentences[i] for i in val_indices]
    test_sentences = [indexed_sentences[i] for i in test_indices]
    
    pad_idx = word2idx[PAD_TOKEN]
    train_ds = NextWordDataset(train_sentences, pad_idx, window_size)
    val_ds = NextWordDataset(val_sentences, pad_idx, window_size)
    test_ds = NextWordDataset(test_sentences, pad_idx, window_size)
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'])
    
    model = NextWordModel(
        vocab_size=len(word2idx),
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        pad_idx=pad_idx,
        rnn_type=rnn_type
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'])
    
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "grad_norms": []}
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        epoch_grad_norms = []
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config['training']['clip_grad_norm'], epoch_grad_norms)
        val_loss, val_acc, val_perp = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["grad_norms"].extend(epoch_grad_norms)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"output/models/best_{rnn_type.lower()}_model.pt")
            
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break
            
    # Final evaluation on test set using BEST model
    model.load_state_dict(torch.load(f"output/models/best_{rnn_type.lower()}_model.pt"))
    test_loss, test_acc, test_perp = evaluate(model, test_loader, criterion, device)
    
    random_baseline = np.log(len(word2idx))
    grad_norm_mean = np.mean(history["grad_norms"]) if history["grad_norms"] else 0.0
    
    # Save result
    result = {
        "model": rnn_type,
        "window_size": window_size,
        "dataset_type": dataset_type,
        "epoch": best_epoch,
        "train_loss": history["train_loss"][best_epoch-1],
        "val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "perplexity": test_perp,
        "random_baseline": random_baseline,
        "gradient_norm_mean": grad_norm_mean
    }
    
    # Plotting
    os.makedirs("output/plots", exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"{rnn_type} Window {window_size} - Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title(f"{rnn_type} Window {window_size} - Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"output/plots/{rnn_type}_{window_size}_{dataset_type}.png")
    plt.close()
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn_type", type=str, default="RNN")
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--dataset_type", type=str, default="short")
    args = parser.parse_args()
    
    res = run_single_experiment(args.rnn_type, args.window_size, args.dataset_type)
    print(res)

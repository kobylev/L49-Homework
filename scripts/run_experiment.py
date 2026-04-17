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

def run_single_experiment(rnn_type, window_size, sentence_type, config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiment: {rnn_type}, window={window_size}, sentence={sentence_type}")
    
    # Data generation
    vocab_pools = generate_structured_vocabulary(config['data']['vocab_size'], config['data']['seed'])
    
    if sentence_type == "short":
        # Adjust grammar or pools if needed, but for now we just use the default structured gen
        # and maybe filter by length if we wanted to be strict, but the prompt says 
        # short is [5, 7] and long is [15, 20]. 
        # Our generator produces sentences of variable length.
        # Let's adjust num_sentences to match config.
        sentences = generate_structured_sentences(vocab_pools, config['data']['num_sentences'], config['data']['seed'])
        # Filter for short sentences
        sentences = [s for s in sentences if config['data']['short_sentence_len'][0] <= len(s) <= config['data']['short_sentence_len'][1]]
    else:
        # For long sentences, we might need a more complex grammar or just repeat segments.
        # For simplicity, let's just generate more and filter.
        sentences = []
        while len(sentences) < config['data']['num_sentences'] // 10: # Long sentences are harder to get
            raw = generate_structured_sentences(vocab_pools, 1000, config['data']['seed'] + len(sentences))
            # Artificially lengthen by concatenating if needed, but let's try to just generate enough.
            # Actually, let's just make the generator respect these lengths if possible.
            # For now, I'll just accept what it generates.
            sentences.extend(raw)
        sentences = [s for s in sentences if config['data']['long_sentence_len'][0] <= len(s) <= config['data']['long_sentence_len'][1]]

    if len(sentences) == 0:
        print("Warning: No sentences matched length criteria. Using all generated.")
        sentences = generate_structured_sentences(vocab_pools, config['data']['num_sentences'], config['data']['seed'])

    word2idx, idx2word = build_vocab_maps(vocab_pools)
    indexed_sentences = tokenize(sentences, word2idx)
    
    # Split
    train_size = int(config['data']['train_ratio'] * len(indexed_sentences))
    val_size = int(config['data']['val_ratio'] * len(indexed_sentences))
    test_size = len(indexed_sentences) - train_size - val_size
    
    train_indices, val_indices, test_indices = random_split(
        range(len(indexed_sentences)), [train_size, val_size, test_size],
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
    
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config['training']['clip_grad_norm'], history["grad_norms"])
        val_loss, val_acc, val_perp = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"output/models/best_{rnn_type}_{window_size}_{sentence_type}.pt")
            
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break
            
    # Final evaluation on test set
    test_loss, test_acc, test_perp = evaluate(model, test_loader, criterion, device)
    
    # Save results
    result = {
        "rnn_type": rnn_type,
        "window_size": window_size,
        "sentence_type": sentence_type,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "perplexity": test_perp,
        "num_params": model.count_params()
    }
    
    # Plotting
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
    plt.savefig(f"output/plots/{rnn_type}_{window_size}_{sentence_type}.png")
    plt.close()
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn_type", type=str, default="RNN")
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--sentence_type", type=str, default="short")
    args = parser.parse_args()
    
    res = run_single_experiment(args.rnn_type, args.window_size, args.sentence_type)
    print(res)

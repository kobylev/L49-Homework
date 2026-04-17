import pandas as pd
from scripts.run_experiment import run_single_experiment
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_plot_a():
    """Accuracy vs Sentence Length (bins) - RNN vs LSTM"""
    # Use window_size 2 for comparison as it's common to both long and short experiments
    stats_dir = "output/stats"
    rnn_long = pd.read_csv(f"{stats_dir}/acc_by_len_RNN_2_long.csv", index_col=0)
    lstm_long = pd.read_csv(f"{stats_dir}/acc_by_len_LSTM_2_long.csv", index_col=0)
    
    # Bins: 5-7, 8-10, 11-13, 14-16, 17-20 (and up to 22)
    bins = [(5, 7), (8, 10), (11, 13), (14, 16), (17, 22)]
    labels = ["5-7", "8-10", "11-13", "14-16", "17-22"]
    
    def get_binned_acc(df):
        df.columns = ["acc"]
        res = []
        for start, end in bins:
            mask = (df.index >= start) & (df.index <= end)
            if mask.any():
                res.append(df[mask]["acc"].mean())
            else:
                res.append(0)
        return res

    rnn_accs = get_binned_acc(rnn_long)
    lstm_accs = get_binned_acc(lstm_long)
    
    plt.figure(figsize=(10, 6))
    plt.plot(labels, rnn_accs, marker='o', label="RNN (Window 2)")
    plt.plot(labels, lstm_accs, marker='s', label="LSTM (Window 2)")
    plt.title("Accuracy vs Sentence Length (Research Phase)")
    plt.xlabel("Sentence Length Bins")
    plt.ylabel("Top-1 Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("output/plots/accuracy_vs_length.png")
    plt.close()

def generate_plot_b():
    """Gradient Norm Heatmap - RNN on long sentences"""
    stats_dir = "output/stats"
    file_path = f"{stats_dir}/grads_RNN_2_long.csv"
    if not os.path.exists(file_path): return
    
    df = pd.read_csv(file_path, index_col=0)
    # df has columns: embedding, rnn, fc
    # We want Epochs on X, Layers on Y
    data = df.T # Rows: Layers, Cols: Epochs
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, annot=True, cmap="YlGnBu", fmt=".4f")
    plt.title("Gradient Norm Heatmap (RNN Long Sentences)")
    plt.xlabel("Training Epoch")
    plt.ylabel("Network Layer")
    plt.tight_layout()
    plt.savefig("output/plots/gradient_heatmap.png")
    plt.close()

def run_all():
    experiments = [
        ("RNN", 1, "short"),
        ("RNN", 2, "short"),
        ("RNN", 3, "short"),
        ("LSTM", 1, "short"),
        ("LSTM", 2, "short"),
        ("LSTM", 3, "short"),
        ("LSTM", 2, "long"),
        ("RNN", 2, "long")
    ]
    
    results = []
    for rnn_type, window_size, dataset_type in experiments:
        res = run_single_experiment(rnn_type, window_size, dataset_type)
        results.append(res)
        
    df = pd.DataFrame(results)
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/results.csv", index=False)
    
    print("\nGenerating Research Visualizations...")
    generate_plot_a()
    generate_plot_b()
    
    print("\nAll experiments completed. Results saved to output/results.csv")
    print(df)

if __name__ == "__main__":
    run_all()

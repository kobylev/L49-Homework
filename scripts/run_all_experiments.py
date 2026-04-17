import pandas as pd
from scripts.run_experiment import run_single_experiment
import os

def run_all():
    experiments = [
        ("RNN", 1, "short"),
        ("RNN", 2, "short"),
        ("RNN", 3, "short"),
        ("LSTM", 1, "short"),
        ("LSTM", 2, "short"),
        ("LSTM", 3, "short"),
        ("LSTM", 2, "long")
    ]
    
    results = []
    for rnn_type, window_size, dataset_type in experiments:
        res = run_single_experiment(rnn_type, window_size, dataset_type)
        results.append(res)
        
    df = pd.DataFrame(results)
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/results.csv", index=False)
    print("\nAll experiments completed. Results saved to output/results.csv")
    print(df)

if __name__ == "__main__":
    run_all()

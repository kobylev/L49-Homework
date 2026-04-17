# RNN Next-Word Prediction Homework

This project explores RNN and LSTM architectures for next-word prediction on a structured synthetic dataset.

## Project Structure

```
L49-Homework/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml          # Central configuration
├── src/
│   ├── dataset.py           # Sliding window dataset
│   ├── preprocessing.py     # Structured grammar & vocab generation
│   ├── model.py             # RNN/LSTM model definitions
│   ├── train.py             # Training logic & Early Stopping
│   └── evaluate.py          # Evaluation metrics
├── scripts/
│   ├── run_experiment.py    # Run a single experiment
│   └── run_all_experiments.py # Run the full suite
├── tests/                   # Comprehensive pytest suite
├── notebooks/               # For post-run analysis
└── output/
    ├── models/              # Saved .pt checkpoints
    ├── plots/               # Loss and Accuracy curves
    └── results.csv          # Consolidated metrics
```

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Tests**:
   ```bash
   $env:PYTHONPATH="."; pytest tests/ -v
   ```

3. **Run All Experiments**:
   ```bash
   $env:PYTHONPATH="."; python scripts/run_all_experiments.py
   ```

## Motivation: Fixing Overfitting

The original project used a purely random dataset, making it impossible for the model to learn meaningful transitions. This resulted in near-zero accuracy and high test loss.

**Solution**: We introduced a structured grammar generator in `src/preprocessing.py`. It uses word clusters (themes) to create predictable patterns (e.g., specific subjects often appearing with specific verbs). This allows the RNN to learn sequential dependencies, bringing test loss down significantly.

## Key Findings

- **RNN vs. LSTM**: LSTMs generally handle longer sequences and larger windows better due to the gating mechanism which mitigates vanishing gradients.
- **Window Size**: Increasing the window size allows the model to capture more context, but requires more data and training time to generalize.
- **Structured Data**: Sequential models require some level of statistical regularity in the data to learn effectively.

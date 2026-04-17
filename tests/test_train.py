import torch
import torch.nn as nn
from src.model import NextWordModel
from src.train import train_one_epoch, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset

def test_early_stopping_triggers():
    es = EarlyStopping(patience=2)
    # Val loss improves
    assert es(1.0) == False
    # Val loss stalls
    assert es(1.0) == False
    assert es(1.1) == True

def test_loss_decreases_over_epochs():
    # Overfit a single batch
    vocab_size = 50
    model = NextWordModel(vocab_size, 16, 32, 1, 0.0, 0, "RNN")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    inputs = torch.randint(1, vocab_size, (1, 5))
    targets = torch.tensor([10])
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=1)
    
    initial_loss = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"), 5.0)
    for _ in range(20):
        final_loss = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"), 5.0)
    
    assert final_loss < initial_loss

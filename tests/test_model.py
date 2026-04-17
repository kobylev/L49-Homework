import torch
from src.model import NextWordModel

def test_rnn_output_shape():
    batch_size = 8
    seq_len = 5
    vocab_size = 100
    model = NextWordModel(vocab_size, 16, 32, 1, 0.1, 0, "RNN")
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, _ = model(x)
    assert logits.shape == (batch_size, vocab_size)

def test_lstm_vs_rnn_same_interface():
    batch_size = 8
    seq_len = 5
    vocab_size = 100
    rnn = NextWordModel(vocab_size, 16, 32, 1, 0.1, 0, "RNN")
    lstm = NextWordModel(vocab_size, 16, 32, 1, 0.1, 0, "LSTM")
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits_rnn, _ = rnn(x)
    logits_lstm, _ = lstm(x)
    
    assert logits_rnn.shape == logits_lstm.shape

def test_gradient_flow():
    vocab_size = 100
    model = NextWordModel(vocab_size, 16, 32, 1, 0.1, 0, "RNN")
    x = torch.randint(0, vocab_size, (8, 5))
    targets = torch.randint(0, vocab_size, (8,))
    logits, _ = model(x)
    loss = torch.nn.CrossEntropyLoss()(logits, targets)
    loss.backward()
    
    assert model.embedding.weight.grad is not None
    assert model.embedding.weight.grad.sum() != 0

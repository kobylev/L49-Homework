import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

class NextWordModel(nn.Module):
    def __init__(self,
                 vocab_size:    int,
                 embedding_dim: int,
                 hidden_dim:    int,
                 num_layers:    int,
                 dropout:       float,
                 pad_idx:       int,
                 rnn_type:      str = "RNN"):
        super().__init__()
        assert rnn_type in ("RNN", "LSTM"), "rnn_type must be 'RNN' or 'LSTM'"

        self.rnn_type   = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        rnn_drop = dropout if num_layers > 1 else 0.0
        shared_kwargs = dict(
            input_size  = embedding_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = rnn_drop,
        )
        if rnn_type == "RNN":
            self.rnn = nn.RNN(**shared_kwargs, nonlinearity="tanh")
        else:
            self.rnn = nn.LSTM(**shared_kwargs)

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                if self.rnn_type == "LSTM":
                    hidden = param.data.size(0) // 4
                    param.data[hidden:2 * hidden].fill_(1.0)

        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits, hidden

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

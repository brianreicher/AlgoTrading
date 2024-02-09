import torch
import torch.nn as nn
import torch.optim as optim


class LTSMCell(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super(LTSMCell, self).__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.W_ih = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.init_weights()

    def init_weights(self) -> None:
        std: float = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            nn.init.uniform_(weight, -std, std)

    def forward(self, input, hx=None) -> tuple[torch.Tensor]:
        if hx is None:
            hx: tuple[torch.Tensor, torch.Tensor] = (torch.zeros(input.size(0), self.hidden_size),
                  torch.zeros(input.size(0), self.hidden_size))
        h, c = hx
        gates = input.matmul(self.W_ih) + h.matmul(self.W_hh) + self.b_ih + self.b_hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate: torch.Tensor = torch.sigmoid(ingate)
        forgetgate: torch.Tensor = torch.sigmoid(forgetgate)
        cellgate: torch.Tensor = torch.tanh(cellgate)
        outgate: torch.Tensor = torch.sigmoid(outgate)
        c: torch.Tensor = (forgetgate * c) + (ingate * cellgate)
        h: torch.Tensor = outgate * torch.tanh(c)
        return h, c

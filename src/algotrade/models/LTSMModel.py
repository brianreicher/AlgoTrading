import torch
import torch.nn as nn
import torch.optim as optim
from .LTSMCell import LTSMCell


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size) -> None:
        super(StockLSTM, self).__init__()
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.cells: nn.ModuleList = nn.ModuleList([LTSMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.fc: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> torch.Tensor:
        h, c = torch.zeros(x.size(0), self.hidden_size), torch.zeros(x.size(0), self.hidden_size)
        for cell in self.cells:
            h, c = cell(x, (h, c))
        out = self.fc(h)
        return out

import torch
import torch.nn as nn
import torch.optim as optim
from .LTSMCell import LTSMCell

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size) -> None:
        """
        Initialize the model 
        Args:
            input_size (int): size of the input features.
            hidden_size (int): size of the hidden state.
            num_layers (int): num of LSTM layers.
            output_size (int): size of the output.

        """
        super(StockLSTM, self).__init__()
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        # Initialize LSTM cells
        self.cells: nn.ModuleList = nn.ModuleList([LTSMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        # Fully connected layer for output
        self.fc: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            torch.Tensor: output tensor.

        """
        # Initialize hidden and cell states
        h, c = torch.zeros(x.size(0), self.hidden_size), torch.zeros(x.size(0), self.hidden_size)
        # Iterate through LSTM cells
        for cell in self.cells:
            # Forward through LSTM cell
            h, c = cell(x, (h, c))
        # Pass the final hidden state through the fully connected layer
        out: torch.Tensor = self.fc(h)
        return out

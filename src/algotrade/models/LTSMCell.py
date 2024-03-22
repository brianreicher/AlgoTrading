import torch
import torch.nn as nn
import torch.optim as optim

class LTSMCell(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        """
        Initialize the cell
        Args:
            input_size (int): size of the input features
            hidden_size (int): size of the hidden state
        """
        super(LTSMCell, self).__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        # Parameters for input-to-hidden and hidden-to-hidden connections
        self.W_ih = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        # Bias terms
        self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        # Initialize weights
        self.init_weights()

    def init_weights(self) -> None:
<<<<<<< HEAD
        std: float = 1.0 / (self.hidden_size**0.5)
=======
        """
        Initialize weights with uniform distribution.
        """
        std: float = 1.0 / (self.hidden_size ** 0.5)
>>>>>>> 630d4844257b87f650420e12304677e36841ca98
        for weight in self.parameters():
            nn.init.uniform_(weight, -std, std)

    def forward(self, input, hx=None) -> tuple[torch.Tensor]:
        """
        Forward pass of the LSTM cell.
        Args:
            input (torch.Tensor): input tensor
            hx (tuple[torch.Tensor, torch.Tensor]): hidden and cell states. Default is None.
        Returns:
            tuple[torch.Tensor]: hidden and cell states.
        """
        # Initialize hidden and cell states 
        if hx is None:
            hx: tuple[torch.Tensor, torch.Tensor] = (
                torch.zeros(input.size(0), self.hidden_size),
                torch.zeros(input.size(0), self.hidden_size),
            )
        h, c = hx
        # Compute gates
        gates = input.matmul(self.W_ih) + h.matmul(self.W_hh) + self.b_ih + self.b_hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        # Apply activation functions
        ingate: torch.Tensor = torch.sigmoid(ingate)
        forgetgate: torch.Tensor = torch.sigmoid(forgetgate)
        cellgate: torch.Tensor = torch.tanh(cellgate)
        outgate: torch.Tensor = torch.sigmoid(outgate)
        # Compute new cell state and hidden state
        c: torch.Tensor = (forgetgate * c) + (ingate * cellgate)
        h: torch.Tensor = outgate * torch.tanh(c)
        return h, c


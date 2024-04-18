import torch
import torch.nn as nn
import numpy as np


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
    def save_checkpoint(self, checkpoint_path):
        torch.save({
            'model_state_dict': self.state_dict(),
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])

def plot_loss_live(writer, epoch, loss):
    writer.add_scalar('Loss/train', loss, epoch)

def predict_metrics(model, input_data, num_future_months, min_vals, max_vals) -> np.ndarray:
    predicted_metrics = []
    for _ in range(num_future_months):
        with torch.no_grad():
            predicted = model(input_data)
            predicted_metrics.append(predicted.squeeze().numpy() * (max_vals - min_vals) + min_vals)
            input_data = torch.cat((input_data[:, 1:, :], predicted.unsqueeze(1)), dim=1)

    return np.array(predicted_metrics)

# set model params
input_size = 5
hidden_size = 64
num_layers = 2
output_size = 5
model = LSTM(input_size, hidden_size, num_layers, output_size)
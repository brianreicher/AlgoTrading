#%%
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


dates =  np.load("healthcare_training_dates.npz")["arr_0"]
dates = np.array([np.datetime64(date) for date in dates])
metrics = np.load("healthcare_training_metrics.npz")["arr_0"]

def average_metric_arrays(dates, nested_arrays)-> list:
    date_to_arrays = defaultdict(list)

    for date, array in zip(dates, nested_arrays):
        date_to_arrays[date].append(array)

    averaged_results = []
    for date, arrays in date_to_arrays.items():
        averaged_array = [sum(x) / len(x) for x in zip(*arrays)]
        averaged_results.append(averaged_array)

    return averaged_results

metrics = np.array(average_metric_arrays(dates, metrics))

min_vals = np.min(metrics, axis=0)
max_vals = np.max(metrics, axis=0)
metrics = (metrics - min_vals) / (max_vals - min_vals)


metrics_tensor = torch.tensor(metrics, dtype=torch.float32)

seq_length = 10

X, y = [], []
for i in range(len(metrics) - seq_length):
    X.append(metrics_tensor[i:i+seq_length])
    y.append(metrics_tensor[i+seq_length])
X = torch.stack(X)
y = torch.stack(y)


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


def plot_future_data(dates, predicted_metrics, n_months, sector):
    plt.figure(figsize=(10, 6))
    plt.title(f"Predicted {sector} Metrics for the Next {n_months} Months")
    plt.xlabel("Date")
    plt.ylabel("Metrics")
    metrics = ["open", "high", "low", "close"]
    for i in range(predicted_metrics.shape[1]-1):
        plt.plot(dates[-1] + np.arange(1, n_months +1, 30), predicted_metrics[:, i], label=metrics[i])
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

input_size = 5
hidden_size = 64
num_layers = 2
output_size = 5
model = LSTM(input_size, hidden_size, num_layers, output_size)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
writer = SummaryWriter()


num_epochs = 301
batch_size = 2
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

train: bool = False

if train:
    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        plot_loss_live(writer, epoch, loss.item())

        if epoch%100==0:
            model.save_checkpoint(f"./LTSM_checkpoint_healthcare_{epoch}")

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.7f}')

n_months = 12*3
# model.load_checkpoint("./LTSM_300") # tech
# model.load_checkpoint("./LTSM_checkpoint_finance_300") # finance
model.load_checkpoint("./LTSM_checkpoint_healthcare_300")

# predict future metrics
input_data = metrics_tensor[-seq_length:].reshape(1, seq_length, input_size)
print(input_data, input_data.shape)
predicted_metrics = predict_metrics(model, input_data, n_months, min_vals, max_vals)
print(f"Predicted metrics for the past {n_months} month(s):\n")
plot_future_data(dates, predicted_metrics, n_months, "Healthcare")

# predict past metrics
input_data = metrics_tensor[-22:-12].reshape(1, seq_length, input_size)
print(input_data, input_data.shape)
predicted_metrics = predict_metrics(model, input_data, 12, min_vals, max_vals)
print(f"Predicted metrics for the past year:\n")
plot_future_data(dates, predicted_metrics, n_months, "Healthcare")

writer.close()


# %%

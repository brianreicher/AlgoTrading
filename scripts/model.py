#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


dates =  np.load("training_dates.npz")["arr_0"]
dates = np.array([np.datetime64(date) for date in dates])
metrics = np.load("training_metrics.npz")["arr_0"]


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


input_size = 5
hidden_size = 64
num_layers = 2
output_size = 5
model = LSTM(input_size, hidden_size, num_layers, output_size)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 50
batch_size = 4
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # if (epoch+1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.7f}')


input_data = metrics_tensor[-seq_length:].reshape(1, seq_length, input_size)
with torch.no_grad():
    predicted = model(input_data)
    predicted_metrics = predicted.squeeze().numpy() * (max_vals - min_vals) + min_vals

print("Predicted metrics for the next month:", predicted_metrics)

# %%

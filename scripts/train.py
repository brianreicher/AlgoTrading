import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from eval import validate
from visualize import *
from process import *
from LSTM import LSTM, plot_loss_live, model


# parse input data files and normalize
dates, metrics = parse_dates_metrics("finance_training_dates.npz", "finance_training_metrics.npz")
min_vals = np.min(metrics, axis=0)
max_vals = np.max(metrics, axis=0)
metrics = (metrics - min_vals) / (max_vals - min_vals)


metrics_tensor = torch.tensor(metrics, dtype=torch.float32)

# set sequence length and convert metrics to LSTM sequence
seq_length = 10
X, y = [], []
for i in range(len(metrics) - seq_length):
    X.append(metrics_tensor[i:i+seq_length])
    y.append(metrics_tensor[i+seq_length])
X = torch.stack(X)
y = torch.stack(y)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
writer = SummaryWriter()


num_epochs = 1001
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
            model.save_checkpoint(f"./LTSM_checkpoint_finance_{epoch}")

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.7f}')

writer.close()

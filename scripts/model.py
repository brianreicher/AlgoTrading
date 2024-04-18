#%%
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torchsummary import summary


def average_metric_arrays(dates, nested_arrays)-> list:
    date_to_arrays = defaultdict(list)

    for date, array in zip(dates, nested_arrays):
        date_to_arrays[date].append(array)

    averaged_results = []
    for date, arrays in date_to_arrays.items():
        averaged_array = [sum(x) / len(x) for x in zip(*arrays)]
        averaged_results.append(averaged_array)

    return averaged_results

def parse_dates_metrics(dates_file, metrics_file)-> tuple:
    dates =  np.load(dates_file)["arr_0"]
    dates = np.array([np.datetime64(date) for date in dates])
    metrics = np.load(metrics_file)["arr_0"]
    metrics = np.array(average_metric_arrays(dates, metrics))
    return dates, metrics



dates, metrics = parse_dates_metrics("finance_training_dates.npz", "finance_training_metrics.npz")
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
    plt.xlabel("Months Forward")
    plt.ylabel("Metrics ($)")
    metrics = ["open", "high", "low", "close"]
    for i in range(predicted_metrics.shape[1]-1):
        plt.plot(np.arange(1, n_months+1), predicted_metrics[:, i], label=metrics[i])
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

#%%
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
            model.save_checkpoint(f"./LTSM_checkpoint_finance_{epoch}")

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.7f}')

n_months = 12*4

def plot_metrics_ytd(ms, sector) -> None:
    metric_names = ["open", "high", "low", "close"]

    plt.figure(figsize=(10, 6))

    for i in range(len(ms[1])-1):
        plt.plot(np.arange(1, 12+1), ms[:, i], label=metric_names[i])
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.xlabel("Past 12 Months")
    plt.ylabel("Metric Values ($)")
    plt.title(f"YTD {sector} Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def validate(actual, predicted):
    mae_past = mean_absolute_error(actual, predicted)
    mse_past = mean_squared_error(actual, predicted)
    rmse_past = np.sqrt(mse_past)
    r2_past = r2_score(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    print("Metrics for Predicted Past YTD:")
    print("MAE:", mae_past)
    print("MSE:", mse_past)
    print("RMSE:", rmse_past)
    print("Mean Absolute Percentage Error:", mape)
    print("R-squared:", r2_past)

dates, metrics = parse_dates_metrics("training_dates.npz", "training_metrics.npz")
min_vals = np.min(metrics, axis=0)
max_vals = np.max(metrics, axis=0)
model.load_checkpoint("./LTSM_checkpoint_tech_300") # tech
input_data = metrics_tensor[-seq_length:].reshape(1, seq_length, input_size)
predicted_future_metrics = predict_metrics(model, input_data, n_months, min_vals, max_vals)
print(f"Predicted metrics for the past {n_months} month(s):\n")
plot_future_data(dates, predicted_future_metrics, n_months, "Tech")
ytd = metrics[-12:]
plot_metrics_ytd(ytd, "Tech")
# predicted past YTD metrics
input_data = metrics_tensor[-22:-12].reshape(1, seq_length, input_size)
predicted_past_metrics = predict_metrics(model, input_data, 12, min_vals, max_vals)
plot_metrics_ytd(predicted_past_metrics, "Tech Backtest YTD")
validate(ytd, predicted_past_metrics)

dates, metrics = parse_dates_metrics("healthcare_training_dates.npz", "healthcare_training_metrics.npz")
min_vals = np.min(metrics, axis=0)
max_vals = np.max(metrics, axis=0)
model.load_checkpoint("./LTSM_checkpoint_healthcare_300")
input_data = metrics_tensor[-seq_length:].reshape(1, seq_length, input_size)
predicted_future_metrics = predict_metrics(model, input_data, n_months, min_vals, max_vals)
print(f"Predicted metrics for the past {n_months} month(s):\n")
plot_future_data(dates, predicted_future_metrics, n_months, "Healthcare")
ytd = metrics[-12:]
plot_metrics_ytd(ytd, "Healthcare")
# predicted past YTD metrics
input_data = metrics_tensor[-22:-12].reshape(1, seq_length, input_size)
predicted_past_metrics = predict_metrics(model, input_data, 12, min_vals, max_vals)
plot_metrics_ytd(predicted_past_metrics, "Healthcare Backtest YTD")
validate(ytd, predicted_past_metrics)

dates, metrics = parse_dates_metrics("finance_training_dates.npz", "finance_training_metrics.npz")
min_vals = np.min(metrics, axis=0)
max_vals = np.max(metrics, axis=0)
model.load_checkpoint("./LTSM_checkpoint_finance_300") # finance
input_data = metrics_tensor[-seq_length:].reshape(1, seq_length, input_size)
# future metrics
predicted_future_metrics = predict_metrics(model, input_data, n_months, min_vals, max_vals)
print(f"Predicted metrics for the past {n_months} month(s):\n")
plot_future_data(dates, predicted_future_metrics, n_months, "Finance")
ytd = metrics[-12:]
plot_metrics_ytd(ytd, "Finance")
# predicted past YTD metrics
input_data = metrics_tensor[-22:-12].reshape(1, seq_length, input_size)
predicted_past_metrics = predict_metrics(model, input_data, 12, min_vals, max_vals)
plot_metrics_ytd(predicted_past_metrics, "Finance Backtest YTD")
validate(ytd, predicted_past_metrics)

writer.close()


# %%

#%%
import numpy as np
import torch
from eval import validate
from visualize import *
from process import *
from LSTM import  predict_metrics, input_size
from train import model, seq_length

n_months = 12*4


dates, metrics = parse_dates_metrics("training_dates.npz", "training_metrics.npz")
min_vals = np.min(metrics, axis=0)
max_vals = np.max(metrics, axis=0)
metrics = torch.tensor(metrics, dtype=torch.float32)
model.load_checkpoint("./LTSM_checkpoint_tech_300") # tech
input_data = metrics[-seq_length:].reshape(1, seq_length, input_size)
predicted_future_metrics = predict_metrics(model, input_data, n_months, min_vals, max_vals)
print(f"Predicted metrics for the past {n_months} month(s):\n")
plot_future_data(dates, predicted_future_metrics, n_months, "Tech")
ytd = metrics[-12:]
plot_metrics_ytd(ytd, "Tech")
# predicted past YTD metrics
input_data = metrics[-22:-12].reshape(1, seq_length, input_size)
predicted_past_metrics = predict_metrics(model, input_data, 12, min_vals, max_vals)
plot_metrics_ytd(predicted_past_metrics, "Tech Backtest YTD")
validate(ytd, predicted_past_metrics)

dates, metrics = parse_dates_metrics("healthcare_training_dates.npz", "healthcare_training_metrics.npz")
min_vals = np.min(metrics, axis=0)
max_vals = np.max(metrics, axis=0)
metrics = torch.tensor(metrics, dtype=torch.float32)
model.load_checkpoint("./LTSM_checkpoint_healthcare_300")
input_data = metrics[-seq_length:].reshape(1, seq_length, input_size)
predicted_future_metrics = predict_metrics(model, input_data, n_months, min_vals, max_vals)
print(f"Predicted metrics for the past {n_months} month(s):\n")
plot_future_data(dates, predicted_future_metrics, n_months, "Healthcare")
ytd = metrics[-12:]
plot_metrics_ytd(ytd, "Healthcare")
# predicted past YTD metrics
input_data = metrics[-22:-12].reshape(1, seq_length, input_size)
predicted_past_metrics = predict_metrics(model, input_data, 12, min_vals, max_vals)
plot_metrics_ytd(predicted_past_metrics, "Healthcare Backtest YTD")
validate(ytd, predicted_past_metrics)

dates, metrics = parse_dates_metrics("finance_training_dates.npz", "finance_training_metrics.npz")
min_vals = np.min(metrics, axis=0)
max_vals = np.max(metrics, axis=0)
metrics = torch.tensor(metrics, dtype=torch.float32)
model.load_checkpoint("./LTSM_checkpoint_finance_1000") # finance
input_data = metrics[-seq_length:].reshape(1, seq_length, input_size)
# future metrics
predicted_future_metrics = predict_metrics(model, input_data, n_months, min_vals, max_vals)
print(f"Predicted metrics for the past {n_months} month(s):\n")
plot_future_data(dates, predicted_future_metrics, n_months, "Finance")
ytd = metrics[-12:]
plot_metrics_ytd(ytd, "Finance")
# predicted past YTD metrics
input_data = metrics[-22:-12].reshape(1, seq_length, input_size)
predicted_past_metrics = predict_metrics(model, input_data, 12, min_vals, max_vals)
plot_metrics_ytd(predicted_past_metrics, "Finance Backtest YTD")
validate(ytd, predicted_past_metrics)


# %%

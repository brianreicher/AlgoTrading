from matplotlib import pyplot as plt
import numpy as np


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
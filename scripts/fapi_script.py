import requests
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()
key: str = os.environ.get("ALPHAVANTAGE_KEY")


def fetch_monthly_adjusted(ticker: str = "IBM") -> dict:
    url: str = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={ticker}&apikey={key}"
    r: requests.Response = requests.get(url)
    response_data: dict = r.json()
    return response_data["Monthly Time Series"]


def get_ticker_data(
    time_series: dict = None,
) -> tuple[list, dict]:
    dates: list = list(time_series.keys())

    metric_names: list[str] = ["open", "high", "low", "close", "volume"]

    metric_dict: dict = {}

    for i, metric_name in enumerate(metric_names):
        key: str = f"{metric_name}s"
        values: list[float] = [
            float(time_series[date][f"{i+1}. {metric_name}"]) for date in dates
        ]
        metric_dict[key] = values

    return dates, metric_dict


def plot_ticker(metrics: tuple = None) -> None:
    plt.figure(figsize=(15, 10))
    dates, vals = metrics[0], metrics[1]

    plt.plot(dates, vals["opens"], label="Open", marker="o")
    plt.plot(dates, vals["highs"], label="High", marker="o")
    plt.plot(dates, vals["lows"], label="Low", marker="o")
    plt.plot(dates, vals["closes"], label="Close", marker="o")
    # plt.bar(dates, vals["volumes"], label="Volume", alpha=0.5)

    plt.xlabel("Date")
    plt.ylabel("Values ($)")
    plt.title(f"Monthly Prices and Volumes Over Time: {dates[len(dates)-1]} - {dates[0]}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    data: dict = fetch_monthly_adjusted(ticker="MSFT")

    metrics: tuple[list, dict] = get_ticker_data(data)

    plot_ticker(metrics)

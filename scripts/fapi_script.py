# imports 
import requests
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import csv 

# get api 
load_dotenv()
key: str = os.environ.get("ALPHAVANTAGE_KEY")

def fetch_monthly_adjusted(ticker: str = "IBM") -> dict:
  """
  fetches monthly adjusted stock data for given ticker from Alpha Vantage API 
    args:
        ticker (str): ticker symbol 
    returns:
        dict: adjusted stock data for ticker 
  
  """
  # URL to fetch monthly adjusted stock data 
  url: str = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={ticker}&apikey={key}"
  # GET request to API, parse response data as json 
  r: requests.Response = requests.get(url)
  response_data: dict = r.json()

  return response_data["Monthly Time Series"]


def get_ticker_data(
    time_series: dict = None,
    ) -> tuple[list, dict]:
    """
    extracts dates and metrics from time series data 
      args: 
        time_series (dict): time series data containing stock metrics
      returns: 
        tuple: tuple containing a list of dates and a dictionary of metric values
    """

    # extract dates, metrics from the time series data
    dates: list = list(time_series.keys())
    metric_names: list[str] = ["open", "high", "low", "close", "volume"]

    # initialize a dictionary to store extracted metric values
    metric_dict: dict = {}

    # iterate over each metric and extract its values for each date
    for i, metric_name in enumerate(metric_names):
        key: str = f"{metric_name}s"
        values: list[float] = [
            float(time_series[date][f"{i+1}. {metric_name}"]) for date in dates
        ]
        metric_dict[key] = values[::-1]
    return dates[::-1], metric_dict


def plot_ticker(metrics: tuple = None) -> None:
    plt.figure(figsize=(15, 10))
    # Extract dates and metric values from the input tuple
    dates, vals = metrics[0], metrics[1]

    # Plot each metric over time with specified label and marker
    plt.plot(dates, vals["opens"], label="Open", marker="o")
    plt.plot(dates, vals["highs"], label="High", marker="o")
    plt.plot(dates, vals["lows"], label="Low", marker="o")
    plt.plot(dates, vals["closes"], label="Close", marker="o")
    # Optionally, plot volume as a bar chart
    # plt.bar(dates, vals["volumes"], label="Volume", alpha=0.5)

    # add labels, display 
    plt.xlabel("Date")
    plt.ylabel("Values ($)")
    plt.title(f"Monthly Prices and Volumes Over Time: {dates[len(dates)-1]} - {dates[0]}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()


def overlay_plot_tickers(ax, metrics: tuple = None, ticker: str = None) -> None:
    dates, vals = metrics[0], metrics[1]

    ax.plot(dates, vals["opens"], label="Open", marker="o")
    ax.plot(dates, vals["highs"], label="High", marker="o")
    ax.plot(dates, vals["lows"], label="Low", marker="o")
    ax.plot(dates, vals["closes"], label="Close", marker="o")

    ax.set_xlabel("Date")
    ax.set_ylabel("Values ($)")
    ax.set_title(f"Monthly Prices Over Time: {dates[0]} - {dates[-1]} ({ticker})")
    ax.legend()
    ax.tick_params(rotation=45)
    ax.set_xticks([])


if __name__ == "__main__":
    tickers: list[str] = ["MSFT", "IBM", "NVDA", "TSLA"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Monthly Prices Over Time for Different Tickers")

    for i, ticker in enumerate(tickers):
        data: dict = fetch_monthly_adjusted(ticker=ticker)
        metrics: tuple[list, dict] = get_ticker_data(data)
        row, col = divmod(i, 2)
        overlay_plot_tickers(axes[row, col], metrics, ticker)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# imports
# %%
import requests
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import csv
import numpy as np


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
    response_data: list = r.json()["Monthly Time Series"]

    url: str = f"https://www.alphavantage.co/query?function=EMA&symbol={ticker}&interval=monthly&time_period=10&series_type=open&apikey={key}"
    r: requests.Response = requests.get(url)
    response_data_ema: list = r.json()["Technical Analysis: EMA"]

    url: str = f"https://www.alphavantage.co/query?function=KAMA&symbol={ticker}&interval=monthly&time_period=10&series_type=open&apikey={key}"
    r: requests.Response = requests.get(url)
    response_data_kama: list = r.json()["Technical Analysis: KAMA"]

    url: str = f"https://www.alphavantage.co/query?function=MAMA&symbol={ticker}&interval=monthly&time_period=10&series_type=open&apikey={key}"
    r: requests.Response = requests.get(url)
    response_data_mama: list = r.json()["Technical Analysis: MAMA"]


    dates: list = list(response_data.keys())[::-1]
    metric_names: list[str] = ["open", "high", "low", "close", "volume", "ema", "mama", "kama"]

    metrics: list = []

    for date in dates:
        date_mets: list = []
        for i, metric_name in enumerate(metric_names):
            if metric_name in ["open", "high", "low", "close", "volume"]:
                value: float = float(response_data[date][f"{i+1}. {metric_name}"])
                date_mets.append(value)
            # elif metric_name == "ema":
            #     value: float = float(response_data_ema[date]["EMA"])
            # elif metric_name == "mama":
            #     value: float = float(response_data_mama[date]["MAMA"])
            # elif metric_name == "kama":
            #     value: float = float(response_data_kama[date]["KAMA"])
            # else:
            #     pass
        metrics.append(date_mets)

    return dates, metrics


def calculate_directionality(metrics) -> list:
    """
    Calculate directionality based on monthly open, close, high, low, and volume.

    Returns:
    float: Directionality value.
    """
    directionalities: list = []
    for i, m in enumerate(metrics):
        price_difference: float = m[3] - m[0]
        normalized_difference: float = price_difference / (m[1] - m[2])
        directionality: float = normalized_difference * m[4]
        directionalities.append(directionality)

    return directionalities

def save_to_csv(dates, stock_metrics, filename) -> None:
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        headers: list[str] = ['Date', 'M1', 'M2', 'M3', 'M4', 'M5']
        csv_writer.writerow(headers)
        
        for date, metrics in zip(dates, stock_metrics):
            row = [date] + metrics
            csv_writer.writerow(row)

if __name__ == "__main__":
    tickers: list[str] = ["MSFT"] #, "IBM", "NVDA", "CRM"]

    # for ticker in tickers:
    #     dates, metrics = fetch_monthly_adjusted(ticker=ticker)
    #     np.savez("./date", dates)
    #     np.savez("./metrics", metrics)
        # yVec: list = calculate_directionality(metrics)
    x = np.load("date.npz")["arr_0"]
    print(x, len(x))


# %%

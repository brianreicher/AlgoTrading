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

    # url: str = f"https://www.alphavantage.co/query?function=EMA&symbol={ticker}&interval=monthly&time_period=10&series_type=open&apikey={key}"
    # r: requests.Response = requests.get(url)
    # response_data_ema: list = r.json()["Technical Analysis: EMA"]

    # url: str = f"https://www.alphavantage.co/query?function=KAMA&symbol={ticker}&interval=monthly&time_period=10&series_type=open&apikey={key}"
    # r: requests.Response = requests.get(url)
    # response_data_kama: list = r.json()["Technical Analysis: KAMA"]

    # url: str = f"https://www.alphavantage.co/query?function=MAMA&symbol={ticker}&interval=monthly&time_period=10&series_type=open&apikey={key}"
    # r: requests.Response = requests.get(url)
    # response_data_mama: list = r.json()["Technical Analysis: MAMA"]


    dates: list = list(response_data.keys())[::-1]
    metric_names: list[str] = ["open", "high", "low", "close", "volume"]#, "ema", "mama", "kama"]

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


if __name__ == "__main__":
    tech_stocks: list = ["MSFT", "IBM", "NVDA", "CRM", "AAPL", "GOOGL", "AMZN", "INTC", "CSCO", "ADBE"]
    finance_tickers: list = ["JPM", "BAC", "WFC", "GS", "C", "MS", "AXP", "USB", "BLK", "PNC"]
    healthcare_stocks: list = ["JNJ", "PFE", "UNH", "MRK", "ABBV", "AMGN", "LLY", "BMY", "CVS", "MDT"]
    consumer_goods_stocks: list = ["PG", "KO", "PEP", "MO", "MCD", "NKE", "CL", "KHC", "WMT", "HD"]
    energy_stocks: list = ["XOM", "CVX", "RDS-A", "BP", "TOT", "COP", "SLB", "EOG", "PTR", "KMI"]

    tickers: dict = {"tech": tech_stocks, "finance": finance_stocks, "healthcare":healthcare_stocks, "consumer_goods":consumer_goods_stocks, "energy":energy_stocks}


    for sector, data in tickers:
        full_dates: list = []
        full_metrics: list = []
        for ticker in data:
            dates, metrics = fetch_monthly_adjusted(ticker=ticker)
            full_dates.extend(dates)
            full_metrics.extend(metrics)

        np.savez(f"./{sector}_training_dates", full_dates)
        np.savez(f"./{sector}_training_metrics", full_metrics)




# %%

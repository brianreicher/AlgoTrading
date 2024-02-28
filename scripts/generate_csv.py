import os
import csv
from typing import List, Tuple
import requests
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load env variables
load_dotenv()
API_KEY: str = os.environ.get("ALPHAVANTAGE_KEY")

# Output directory for CSV files
OUTPUT_DIRECTORY = "data/"

def fetch_monthly_adjusted(ticker: str = "IBM") -> dict:
  """
  Fetches monthly adjusted stock data for a given ticker from Alpha Vantage API

  Args:
      ticker (str): Ticker symbol.

  Returns:
      dict: Monthly adjusted stock data for the ticker.
  """
  # URL to fetch monthly adjusted stock data 
  url: str = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={ticker}&apikey={key}"
  # GET request to API, parse response data as json 
  r: requests.Response = requests.get(url)
  response_data: dict = r.json() 
  print("Response data:", response_data) 
  return response_data["Monthly Time Series"]

def get_ticker_data(time_series: dict) -> Tuple[list, dict]:
  """
  Extracts dates and metrics from time series data.

  Args:
      time_series (dict): Time series data containing stock metrics.

  Returns:
      tuple: Tuple containing a list of dates and a dictionary of metric values.
  """
  dates = list(time_series.keys())
  metric_names = ["open", "high", "low", "close", "volume"]
  metric_dict = {}

  for metric_name in metric_names:
      values = [float(time_series[date][f"{i+1}. {metric_name}"]) for i, date in enumerate(dates)]
      metric_dict[f"{metric_name}s"] = values[::-1]

  return dates[::-1], metric_dict

def save_to_csv(ticker: str, dates: list, metric_dict: dict) -> None:
  """
  Save stock data to a CSV file.

  Args:
      ticker (str): Ticker symbol of the stock.
      dates (list): List of dates.
      metric_dict (dict): Dictionary containing metric values.
  """
  file_name = os.path.join(OUTPUT_DIRECTORY, f"{ticker}_monthly_data.csv")
  with open(file_name, mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["Date", "Open", "High", "Low", "Close"])
      writer.writerows(zip(dates, metric_dict["opens"], metric_dict["highs"], metric_dict["lows"], metric_dict["closes"]))

if __name__ == "__main__":
  tickers = ["MSFT", "IBM", "NVDA", "TSLA"]

  for ticker in tickers:
    data = fetch_monthly_adjusted(ticker)
    dates, metric_dict = get_ticker_data(data)
    save_to_csv(ticker, dates, metric_dict)


if __name__ == "__main__":
    # Test fetch_monthly_adjusted function
    ticker_data = fetch_monthly_adjusted("MSFT")
    print("Monthly adjusted data for MSFT:")
    print(ticker_data)

    # Test get_ticker_data function
    dates, metric_dict = get_ticker_data(ticker_data)
    print("Dates:")
    print(dates)
    print("Metric Dictionary:")
    print(metric_dict)

    # Test save_to_csv function
    save_to_csv("MSFT", dates, metric_dict)
    print("Data saved to CSV file.")

import requests
from dotenv import load_dotenv
import os

load_dotenv()

key: str = os.environ.get("ALPHAVANTAGE_KEY")
url: str = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=IBM&apikey={key}"
r: requests.Response = requests.get(url)
data: dict = r.json()

print(data)

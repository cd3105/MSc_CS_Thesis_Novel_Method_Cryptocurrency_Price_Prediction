#This example uses Python 2.7 and the python-request library.

from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
from datetime import datetime
import json
import pandas as pd

url = 'https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical'
parameters_1 = {
  'start':'1',
  'limit':'500',
}
parameters_2 = {
  'start':'501',
  'limit':'500',
}
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': 'beebe912-3c7c-4bf6-87a6-cd9528dc27a9',
}

session = Session()
session.headers.update(headers)

try:
  response = session.get(url, params=parameters_1)
  data_1 = json.loads(response.text)
  df_1 = pd.DataFrame(data_1['data']).rename(columns={'timestamp': 'Time', 'value': 'Fear_and_Greed_Index', 'value_classification': 'Fear_and_Greed_Index_Classification'})
  print(df_1)
except (ConnectionError, Timeout, TooManyRedirects) as e:
  print(e)

try:
  response = session.get(url, params=parameters_2)
  data_2 = json.loads(response.text)
  df_2 = pd.DataFrame(data_2['data']).rename(columns={'timestamp': 'Time', 'value': 'Fear_and_Greed_Index', 'value_classification': 'Fear_and_Greed_Index_Classification'})
  print(df_2)
except (ConnectionError, Timeout, TooManyRedirects) as e:
  print(e)

concat_df = pd.concat([df_1, df_2]).sort_values('Time').reset_index(drop=True)

concat_df.to_csv("All_Crypto_Data/Crypto_F_and_G_Index/CryptoMarketCap/1_Day/" + f"CryptoMarketCap_Daily_Fear_and_Greed_Index_{str(datetime.utcfromtimestamp(int(list(concat_df['Time'])[0])).date()).replace('-', '_')}__{str(datetime.utcfromtimestamp(int(list(concat_df['Time'])[-1])).date()).replace('-', '_')}.csv")

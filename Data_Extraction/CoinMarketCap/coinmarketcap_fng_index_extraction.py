from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
from datetime import datetime
import json
import pandas as pd
import os
from dotenv import load_dotenv

# Script for extracting the Fear and Greed Index using the CoinMarketCap API

load_dotenv()

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
  'X-CMC_PRO_API_KEY': os.getenv("CMC_API_KEY"),
}

session = Session()
session.headers.update(headers)

try:
  response = session.get(url, params=parameters_1)
  data_1 = json.loads(response.text)
  df_1 = pd.DataFrame(data_1['data']).rename(columns={'timestamp': 'TIMESTAMP', 'value': 'FEAR_AND_GREED_INDEX', 'value_classification': 'FEAR_AND_GREED_INDEX_CLASSIFICATION'})
  df_1['TIMESTAMP'] = df_1['TIMESTAMP'].apply(lambda x: datetime.utcfromtimestamp(int(x)))
except (ConnectionError, Timeout, TooManyRedirects) as e:
  print(e)

try:
  response = session.get(url, params=parameters_2)
  data_2 = json.loads(response.text)
  df_2 = pd.DataFrame(data_2['data']).rename(columns={'timestamp': 'TIMESTAMP', 'value': 'FEAR_AND_GREED_INDEX', 'value_classification': 'FEAR_AND_GREED_INDEX_CLASSIFICATION'})
  df_2['TIMESTAMP'] = df_2['TIMESTAMP'].apply(lambda x: datetime.utcfromtimestamp(int(x)))
except (ConnectionError, Timeout, TooManyRedirects) as e:
  print(e)

concat_df = pd.concat([df_1, df_2]).sort_values('TIMESTAMP', ascending=True).reset_index(drop=True)

concat_df.to_csv("All_Crypto_Data/Crypto_F_and_G_Index/CryptoMarketCap/1_Day/" + f"CryptoMarketCap_Daily_Fear_and_Greed_Index_{concat_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{concat_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", index=False)

import os
import requests
import pandas as pd
from datetime import datetime
from io import StringIO

market_chart_dict = {'Market_Price_USD': 'market-price',
                     'Total_Circulating_Coins': 'total-bitcoins',
                     'Market_Cap_USD': 'market-cap',
                     'Exchange_Trade_Volume_USD': 'trade-volume',}

frequency_dict = {'Market_Price_USD': '1_Day',
                  'Total_Circulating_Coins': 'N_Min',
                  'Market_Cap_USD': 'N_Min',
                  'Exchange_Trade_Volume_USD': '1_Day',}

for mk in market_chart_dict.keys():
    current_url = f"https://api.blockchain.info/charts/{market_chart_dict[mk]}?timespan=20years&format=csv&sampled=false"

    current_response = requests.get(current_url)

    if current_response.status_code == 200:
        csv_data = StringIO(current_response.text)
        current_df = pd.read_csv(csv_data, header=None, names=['TIMESTAMP', f'BTC_{mk.upper()}'])

        current_file_path = f'All_Crypto_Data/Crypto_Market_Data/Unmerged/Blockchain/BTC/{frequency_dict[mk]}/'

        if not os.path.exists(current_file_path):
            os.makedirs(current_file_path)

        current_file_name = f'Blockchain_BTC_USD_{mk}_{datetime.strptime(list(current_df["TIMESTAMP"])[0], "%Y-%m-%d %H:%M:%S").strftime("%d_%m_%Y")}__{datetime.strptime(list(current_df["TIMESTAMP"])[-1], "%Y-%m-%d %H:%M:%S").strftime("%d_%m_%Y")}.csv'

        current_df.to_csv(current_file_path + current_file_name, index=False)
    else:
        print(f"Failed to retrieve data: {current_response.status_code} (Chart: {mk})")

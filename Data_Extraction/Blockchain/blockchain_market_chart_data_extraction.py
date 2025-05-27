import requests
import pandas as pd
from datetime import datetime
from io import StringIO

market_chart_dict = {'Market_Price': 'market-price',
                     'Total_Circulating_Coins': 'total-bitcoins',
                     'Market_Cap': 'market-cap',
                     'Exchange_Trade_Volume': 'trade-volume',
                     'Market_Value_to_Realised_Value': 'mvrv',
                     'Network_Value_to_Transactions': 'nvt',
                     'Network_Value_to_Transactions_Signal': 'nvts',
                     'Profitable_Days': 'bitcoin-profitable-days',
                     '2_Year_MA_Multiplier': '2y-moving-average',
                     'Pi_Cycle_Top_Indicator': 'pi-cycle-top-indicator',
                     '200_Week_MA': '200w-moving-avg-heatmap',}

for mk in market_chart_dict.keys():
    current_url = f"https://api.blockchain.info/charts/{market_chart_dict[mk]}?timespan=20years&format=csv&sampled=false"

    current_response = requests.get(current_url)

    if current_response.status_code == 200:
        csv_data = StringIO(current_response.text)
        current_df = pd.read_csv(csv_data, header=None, names=['Time', mk])

        print(current_df)

        current_file_path = 'All_Crypto_Data/Crypto_Market_Data/Blockchain/BTC/'
        current_file_name = f'Blockchain_BTC_USD_{mk}_{datetime.strptime(list(current_df["Time"])[0], "%Y-%m-%d %H:%M:%S").strftime("%d-%m-%Y").replace("-", "_")}__{datetime.strptime(list(current_df["Time"])[-1], "%Y-%m-%d %H:%M:%S").strftime("%d-%m-%Y").replace("-", "_")}.csv'

        current_df.to_csv(current_file_path + current_file_name)
    else:
        print(f"Failed to retrieve data: {current_response.status_code} (Chart: {mk})")

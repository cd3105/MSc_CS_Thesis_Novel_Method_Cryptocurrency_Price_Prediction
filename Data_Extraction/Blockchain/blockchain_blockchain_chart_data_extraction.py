import os
import requests
import pandas as pd
from datetime import datetime
from io import StringIO

blockchain_chart_dict = {'Blockchain_Size_Megabytes': 'blocks-size',
                         'Average_Block_Size_Megabytes': 'avg-block-size',
                         'Average_Transaction_Count_Per_Block': 'n-transactions-per-block',
                         'Total_Transaction_Count': 'n-transactions-total',
                         'Median_Transaction_Confirmation_Time_Minutes': 'median-confirmation-time',
                         'Average_Transaction_Confirmation_Time_Minutes': 'avg-confirmation-time',
                         'Hash_Rate_Terahashes_Per_Second': 'hash-rate',
                         'Mining_Difficulty_Hashes': 'difficulty',
                         'Miners_Revenue_USD': 'miners-revenue',
                         'Total_Transaction_Fees_BTC': 'transaction-fees',
                         'Total_Transaction_Fees_USD': 'transaction-fees-usd',
                         'Cost_Percentage_of_Trade_Volume': 'cost-per-transaction-percent',
                         'Cost_Per_Transaction_USD': 'cost-per-transaction',
                         'Total_Unique_Used_Address_Count': 'n-unique-addresses',
                         'Total_Confirmed_Transaction_Count': 'n-transactions',
                         'Total_Transaction_Count_Per_Second': 'transactions-per-second',
                         'Total_Output_Value_BTC': 'output-volume',
                         'Total_Unconfirmed_Transaction_Count_In_Mempool': 'mempool-count',
                         'Mempool_Size_Growth_Bytes_per_Second': 'mempool-growth',
                         'Mempool_Size_Bytes': 'mempool-size',
                         'Total_Unspent_Transaction_Output_Count': 'utxo-count',
                         'Total_Transaction_Count_Excluding_Top_100_Addresses': 'n-transactions-excluding-popular',
                         'Total_Estimated_Transaction_Value_BTC': 'estimated-transaction-volume',
                         'Total_Estimated_Transaction_Value_USD': 'estimated-transaction-volume-usd',}

frequency_dict = {'Blockchain_Size_Megabytes': '1_Day',
                  'Average_Block_Size_Megabytes': '1_Day',
                  'Average_Transaction_Count_Per_Block': '1_Day',
                  'Total_Transaction_Count': '1_Day',
                  'Median_Transaction_Confirmation_Time_Minutes': '1_Day',
                  'Average_Transaction_Confirmation_Time_Minutes': '6_Hour',
                  'Hash_Rate_Terahashes_Per_Second': '1_Day',
                  'Mining_Difficulty_Hashes': '1_Day',
                  'Miners_Revenue_USD': '1_Day',
                  'Total_Transaction_Fees_BTC': '1_Day',
                  'Total_Transaction_Fees_USD': '1_Day',
                  'Cost_Percentage_of_Trade_Volume': '1_Day',
                  'Cost_Per_Transaction_USD': '1_Day',
                  'Total_Unique_Used_Address_Count': '1_Day',
                  'Total_Confirmed_Transaction_Count': '1_Day',
                  'Total_Transaction_Count_Per_Second': '15_Min',
                  'Total_Output_Value_BTC': '1_Day',
                  'Total_Unconfirmed_Transaction_Count_In_Mempool': '15_Min',
                  'Mempool_Size_Growth_Bytes_per_Second': '15_Min',
                  'Mempool_Size_Bytes': '15_Min',
                  'Total_Unspent_Transaction_Output_Count': 'N_Min',
                  'Total_Transaction_Count_Excluding_Top_100_Addresses': '1_Day',
                  'Total_Estimated_Transaction_Value_BTC': '1_Day',
                  'Total_Estimated_Transaction_Value_USD': '1_Day',}

for bk in blockchain_chart_dict.keys():
    current_url = f"https://api.blockchain.info/charts/{blockchain_chart_dict[bk]}?timespan=20years&format=csv&sampled=false"

    current_response = requests.get(current_url)

    if current_response.status_code == 200:
        csv_data = StringIO(current_response.text)
        current_df = pd.read_csv(csv_data, header=None, names=['TIMESTAMP', f'BTC_{bk.upper()}'])

        current_file_path = f'All_Crypto_Data/Blockchain_Data/Unmerged/Blockchain/BTC/{frequency_dict[bk]}/'

        if not os.path.exists(current_file_path):
            os.makedirs(current_file_path)

        current_file_name = f'Blockchain_BTC_USD_{bk}_{datetime.strptime(list(current_df["TIMESTAMP"])[0], "%Y-%m-%d %H:%M:%S").strftime("%d_%m_%Y")}__{datetime.strptime(list(current_df["TIMESTAMP"])[-1], "%Y-%m-%d %H:%M:%S").strftime("%d_%m_%Y")}.csv'

        current_df.to_csv(current_file_path + current_file_name, index=False)
    else:
        print(f"Failed to retrieve data: {current_response.status_code} (Chart: {bk})")

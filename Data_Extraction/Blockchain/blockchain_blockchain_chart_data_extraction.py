import requests
import pandas as pd
from datetime import datetime
from io import StringIO

blockchain_chart_dict = {'Blockchain_Size_MB': 'blocks-size',
                         'Average_Block_Size_MB': 'avg-block-size',
                         'Average_Transaction_Per_Block': 'n-transactions-per-block',
                         'Average_Payments_Per_Block': 'n-payments-per-block',
                         'Total_Transactions': 'n-transactions-total',
                         'Median_Confirmation_Time': 'median-confirmation-time',
                         'Average_Confirmation_Time': 'avg-confirmation-time',
                         'Hash_Rate': 'hash-rate',
                         'Mining_Difficulty': 'difficulty',
                         'Miners_Revenue_USD': 'miners-revenue',
                         'Total_Transaction_Fees_BTC': 'transaction-fees',
                         'Total_Transaction_Fees_USD': 'transaction-fees-usd',
                         'Average_Fees_Per_Transaction_USD': 'fees-usd-per-transaction',
                         'Cost_Percentage_of_Trade_Volume': 'cost-per-transaction-percent',
                         'Cost_Per_Transaction_USD': 'cost-per-transaction',
                         'Total_Unique_Addresses_Used': 'n-unique-addresses',
                         'Total_Confirmed_Transactions': 'n-transactions',
                         'Total_Confirmed_Payments': 'n-payments',
                         'Total_Transactions_Per_Second': 'transactions-per-second',
                         'Total_Output_Value': 'output-volume',
                         'Total_Unconfirmed_Transactions_In_Mempool': 'mempool-count',
                         'Mempool_Size_Growth_Bs_per_Second': 'mempool-growth',
                         'Mempool_Size_B': 'mempool-size',
                         'Total_Unspent_Transaction_Outputs': 'utxo-count',
                         'Total_Transactions_Excluding_Top_100_Addresses': 'n-transactions-excluding-popular',
                         'Total_Estimated_Transaction_Value_BTC': 'estimated-transaction-volume',
                         'Total_Estimated_Transaction_Value_USD': 'estimated-transaction-volume-usd',}

for bk in blockchain_chart_dict.keys():
    current_url = f"https://api.blockchain.info/charts/{blockchain_chart_dict[bk]}?timespan=20years&format=csv&sampled=false"

    current_response = requests.get(current_url)

    if current_response.status_code == 200:
        csv_data = StringIO(current_response.text)
        current_df = pd.read_csv(csv_data, header=None, names=['Time', bk])

        print(current_df)

        current_file_path = 'All_Crypto_Data/Blockchain_Data/Blockchain/BTC/'
        current_file_name = f'Blockchain_BTC_USD_{bk}_{datetime.strptime(list(current_df["Time"])[0], "%Y-%m-%d %H:%M:%S").strftime("%d-%m-%Y").replace("-", "_")}__{datetime.strptime(list(current_df["Time"])[-1], "%Y-%m-%d %H:%M:%S").strftime("%d-%m-%Y").replace("-", "_")}.csv'

        current_df.to_csv(current_file_path + current_file_name)
    else:
        print(f"Failed to retrieve data: {current_response.status_code} (Chart: {bk})")

import nasdaqdatalink
import pandas as pd
import os
from dotenv import load_dotenv

# Script for extracting BTC Blockchain Data using the NASDAQ API

load_dotenv()

data_code_dict = {"ATRCT": "Median_Transaction_Confirmation_Time_Minutes",
                  "AVBLS": "Average_Block_Size_Megabytes",
                  "BLCHS": "Blockchain_Size_Megabytes",
                  "CPTRA": "Cost_Per_Transaction_USD",
                  "CPTRV": "Cost_Percentage_Transaction_Volume",
                  "DIFF": "Mining_Difficulty_Hashes",
                  "ETRAV": "Estimated_Transaction_Volume_BTC",
                  "ETRVU": "Estimated_Transaction_Volume_USD",
                  "HRATE": "Hash_Rate_Terahashes_Per_Second",
                  "MIREV": "Miners_Revenue_USD",
                  "MWNTD": "My_Wallet_Transaction_Count",
                  "MWNUS": "My_Wallet_User_Count",
                  "MWTRV": "My_Wallet_Transaction_Volume_USD",
                  "NADDU": "Unique_Addresses_Used_Count",
                  "NTRAN": "Total_Confirmed_Transaction_Count",
                  "NTRAT": "Total_Transaction_Count",
                  "NTRBL": "Transaction_Count_Per_Block",
                  "NTREP": "Transaction_Count_Excluding_Popular_Addresses",
                  "TOUTV": "Total_Output_Volume_BTC",
                  "TRFEE": "Total_Transaction_Fees_BTC",
                  "TRFUS": "Total_Transaction_Fees_USD",
                  "BCDDC": "Cumulative_Days_Destroyed",
                  "BCDDE": "Days_Destroyed",
                  "BCDDM": "Minimum_Age_1_Month_Days_Destroyed",
                  "BCDDW": "Minimum_Age_1_Week_Days_Destroyed",
                  "BCDDY": "Minimum_Age_1_Year_Days_Destroyed",
                  "NETDF": "Network_Deficit",
                  "TVTVR": "Trade_Volume_VS_Transaction_Volume_Ratio"}
nasdaqdatalink.ApiConfig.api_key = os.getenv("NASDAQ_API_KEY") 
full_df = pd.DataFrame(columns=['TIMESTAMP'])
file_path = "All_Crypto_Data/Blockchain_Data/Unmerged/Blockchain_via_Quandl/BTC/1_Day/"
total_file_path = "All_Crypto_Data/Blockchain_Data/Merged/Blockchain_via_Quandl/BTC/1_Day/"

if not os.path.exists(file_path):
    os.makedirs(file_path)

for k in data_code_dict.keys():
    current_df = nasdaqdatalink.get_table('QDL/BCHAIN',code=k).drop(['code'], axis=1).rename(columns={'date':'TIMESTAMP', 'value': f'BTC_{data_code_dict[k].upper()}'}).sort_values(by='TIMESTAMP').reset_index(drop=True)
    current_file_name = f"Blockchain_via_Quandl_BTC_Daily_{data_code_dict[k]}_{list(current_df['TIMESTAMP'])[0].strftime('%d-%m-%Y').replace('-', '_')}__{list(current_df['TIMESTAMP'])[-1].strftime('%d-%m-%Y').replace('-', '_')}.csv"
    full_df = pd.merge(full_df, current_df, on='TIMESTAMP', how='outer')

    current_df.to_csv(file_path + current_file_name, index=False)

if not os.path.exists(total_file_path):
    os.makedirs(total_file_path)

full_df.to_csv(f"{total_file_path}Blockchain_via_Quandl_BTC_Daily_Blockchain_Data_{list(full_df['TIMESTAMP'])[0].strftime('%d-%m-%Y').replace('-', '_')}__{list(full_df['TIMESTAMP'])[-1].strftime('%d-%m-%Y').replace('-', '_')}.csv", index=False)

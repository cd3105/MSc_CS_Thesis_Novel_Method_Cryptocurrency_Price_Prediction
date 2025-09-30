import nasdaqdatalink
import pandas as pd
import os
from dotenv import load_dotenv

# Script for extracting BTC Market Data using the NASDAQ API

load_dotenv()

data_code_dict = {"MKPRU": "Market_Price_USD",
                  "MKTCP": "Market_Cap_USD",
                  "TRVOU": "USD_Exchange_Trade_Volume",
                  "TOTBC": "Total_Coins",}

nasdaqdatalink.ApiConfig.api_key = os.getenv("NASDAQ_API_KEY") 
full_df = pd.DataFrame(columns=['TIMESTAMP'])
file_path = "All_Crypto_Data/Crypto_Market_Data/Unmerged/Blockchain_via_Quandl/BTC/1_Day/"
total_file_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain_via_Quandl/BTC/1_Day/"

if not os.path.exists(file_path):
    os.makedirs(file_path)

for k in data_code_dict.keys():
    current_df = nasdaqdatalink.get_table('QDL/BCHAIN',code=k).drop(['code'], axis=1).rename(columns={'date':'TIMESTAMP', 'value':f'BTC_{data_code_dict[k].upper()}'}).sort_values(by='TIMESTAMP').reset_index(drop=True)
    current_file_name = f"Blockchain_via_Quandl_BTC_USD_Daily_{data_code_dict[k]}_{list(current_df['TIMESTAMP'])[0].strftime('%d-%m-%Y').replace('-', '_')}__{list(current_df['TIMESTAMP'])[-1].strftime('%d-%m-%Y').replace('-', '_')}.csv"
    full_df = pd.merge(full_df, current_df, on='TIMESTAMP', how='outer')

    current_df.to_csv(file_path + current_file_name, index=False)

if not os.path.exists(total_file_path):
    os.makedirs(total_file_path)

full_df.to_csv(f"{total_file_path}Blockchain_via_Quandl_BTC_USD_Daily_Market_Data_{list(full_df['TIMESTAMP'])[0].strftime('%d-%m-%Y').replace('-', '_')}__{list(full_df['TIMESTAMP'])[-1].strftime('%d-%m-%Y').replace('-', '_')}.csv", index=False)

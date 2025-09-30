import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Script for retrieving Launch Date information of a large quantity of cryptocurrencies using the CoinDesk API

load_dotenv()

kaggle_23CCs_market_data_base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Kaggle/Kaggle_Daily_USD_Market_Data_23_CCs_until_07_07_2021/"
kaggle_109CCs_market_data_base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Kaggle/Kaggle_Daily_USD_Market_Data_109_CCs_until_14_05_2025/"
kaggle_425CCs_market_data_base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Kaggle/Kaggle_Daily_USD_Market_Data_425_CCs_until_28_09_2022/"
yahoo_finance_market_data_base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Yahoo_Finance/"
launch_date_path = "All_Crypto_Data/Crypto_Launch_Data/"
unique_cryptocurrencies = []
cryptocurrencies = []
launch_dates = []

paths = [kaggle_23CCs_market_data_base_path,
         kaggle_109CCs_market_data_base_path,
         kaggle_425CCs_market_data_base_path,
         yahoo_finance_market_data_base_path,]

for p in paths:
    unique_cryptocurrencies += list([m for m in os.listdir(p) if not m.endswith('.txt')])

unique_cryptocurrencies = sorted(list(set(unique_cryptocurrencies)))

for idx in range(0, len(unique_cryptocurrencies), 50):
    print(f"Current Number of CCs Processed: {idx+50}")

    current_cryptocurrencies = unique_cryptocurrencies[idx:idx+50]
    current_response = requests.get('https://data-api.coindesk.com/asset/v2/metadata',
                                    params={"assets":current_cryptocurrencies,
                                            "api_key":os.getenv("CD_API_KEY")},
                                            headers={"Content-type":"application/json; charset=UTF-8"}
                                            )

    if current_response.status_code == 200:
        current_json = current_response.json()
        current_launch_dates = [(cc, current_json['Data'][cc]['LAUNCH_DATE']) for cc in current_cryptocurrencies if (cc in current_json['Data'].keys()) and ('LAUNCH_DATE' in current_json['Data'][cc].keys())] 
        
        cryptocurrencies += [cc for (cc, ld) in current_launch_dates]
        launch_dates += [ld for (cc, ld) in current_launch_dates]

launch_date_df = pd.DataFrame({'CC':cryptocurrencies, 
                               'LAUNCH_DATE':launch_dates})
launch_date_df = launch_date_df[launch_date_df["LAUNCH_DATE"] != None].dropna()
launch_date_df['LAUNCH_DATE'] = launch_date_df['LAUNCH_DATE'].apply(lambda x: datetime.utcfromtimestamp(x))

if not os.path.exists(launch_date_path):
    os.makedirs(launch_date_path)

launch_date_df.to_csv(f"{launch_date_path}Crypto_Launch_Dates.csv", 
                      index=False)

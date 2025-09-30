import pandas as pd
import datetime
import os

# Script for reordering and renaming files containing Market Data of 425 CCs

map_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Kaggle/Kaggle_Daily_USD_Market_Data_425_CCs_until_28_09_2022/"

for f in [f for f in os.listdir(map_path) if f.endswith(".csv")]:
    file_path = f"{map_path}{f}"
    current_cc = f[:-8]
    current_cc_df = pd.read_csv(file_path).rename(columns={'timestamp':'TIMESTAMP',
                                                           'adjclose':'ADJUSTED_CLOSE_PRICE_USD',
                                                           'open':'OPEN_PRICE_USD',
                                                           'high':'HIGH_PRICE_USD',
                                                           'volume':'VOLUME_USD',
                                                           'low':'LOW_PRICE_USD',
                                                           'close':'CLOSE_PRICE_USD',})

    current_cc_df['TIMESTAMP'] = pd.to_datetime(current_cc_df['TIMESTAMP'])
    current_cc_df = current_cc_df.sort_values(by='TIMESTAMP', ascending=True)
    new_file_name = f"Kaggle_{f[:-8]}_Daily_USD_Market_Data_{current_cc_df.iloc[0]['TIMESTAMP'].strftime('%d_%m_%Y')}__{current_cc_df.iloc[-1]['TIMESTAMP'].strftime('%d_%m_%Y')}.csv"
    new_directory_path = f"{map_path}{f[:-8]}/"
    
    if not os.path.exists(new_directory_path):
        os.mkdir(new_directory_path)
    
    current_cc_df.to_csv(f"{new_directory_path}{new_file_name}", index=False)
        
import pandas as pd
import os
import re
from datetime import datetime

base_path = "All_Crypto_Data/Macroeconomic_Data/Unmerged/Investing/1_Day/"

exchange_rates = ['AUD_USD', 'CAD_USD','CHF_USD','CNY_USD','DKK_USD','EUR_USD','GBP_USD','JPY_USD','NOK_USD','NZD_USD','RUB_USD','SEK_USD','SGD_USD']

merged_df = pd.DataFrame(columns={'TIMESTAMP':[]})

for f in os.listdir(base_path):
    current_prefix = re.search(r'Investing_(.*?)_Daily', f).group(1).upper()

    current_df = pd.read_csv(f"{base_path}{f}").rename(columns={'Date':'TIMESTAMP',
                                                                'Price':f'{current_prefix}_PRICE',
                                                                'Open':f'{current_prefix}_OPEN_PRICE',
                                                                'Low':f'{current_prefix}_LOW_PRICE',
                                                                'High':f'{current_prefix}_HIGH_PRICE',
                                                                'Vol.':f'{current_prefix}_VOLUME',
                                                                'Change %':f'{current_prefix}_CHANGE_PERCENTAGE'})

    current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])
    current_df = current_df.sort_values('TIMESTAMP', 
                                        ascending=True).reset_index(drop=True)
    current_date_range_df = pd.DataFrame({
        'TIMESTAMP': pd.date_range(start=datetime(2009, 1, 1), end=current_df['TIMESTAMP'].max(), freq='D')
    }).sort_values('TIMESTAMP', ascending=True)
    current_df = pd.merge(current_df, current_date_range_df, on='TIMESTAMP', how='right')
    merged_df = pd.merge(merged_df, current_df, on='TIMESTAMP', how='outer').sort_values('TIMESTAMP').reset_index(drop=True)

new_map_path = "All_Crypto_Data/Macroeconomic_Data/Merged/Investing/1_Day/"
file_name = f"Investing_Daily_Macroeconomic_Data_USD_{merged_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"

if not os.path.exists(new_map_path):
    os.makedirs(new_map_path)

merged_df.to_csv(f"{new_map_path}{file_name}", index=False)

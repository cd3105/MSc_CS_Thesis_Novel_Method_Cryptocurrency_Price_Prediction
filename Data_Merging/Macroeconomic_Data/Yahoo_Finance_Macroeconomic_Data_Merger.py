import pandas as pd
import os
import re
from datetime import datetime

# Script to Merge Yahoo Finance Data into a single Dataset

base_path = "All_Crypto_Data/Macroeconomic_Data/Unmerged/Yahoo_Finance/1_Day/"

exchange_rates = ['AUD_USD', 'CAD_USD','CHF_USD','CNY_USD','DKK_USD','EUR_USD','GBP_USD','JPY_USD','NOK_USD','NZD_USD','RUB_USD','SEK_USD','SGD_USD']

merged_df = pd.DataFrame(columns={'TIMESTAMP':[]})

for f in os.listdir(base_path):
    current_prefix = re.search(r'Yahoo_Finance_(.*?)_Daily', f).group(1).upper()

    if current_prefix in exchange_rates:
        current_prefix += '_EXCHANGE_RATE'

    current_df = pd.read_csv(f"{base_path}{f}", index_col=0).dropna().rename(columns={'Date':'TIMESTAMP',
                                                                                      'Open':f'{current_prefix}_OPEN_PRICE',
                                                                                      'Low':f'{current_prefix}_LOW_PRICE',
                                                                                      'High':f'{current_prefix}_HIGH_PRICE',
                                                                                      'Close':f'{current_prefix}_CLOSE_PRICE',
                                                                                      'Volume':f'{current_prefix}_VOLUME',
                                                                                      'Dividends':f'{current_prefix}_DIVIDENDS',
                                                                                      'Stock Splits':f'{current_prefix}_STOCK_SPLITS',})
    current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'], utc=True)
    current_df['TIMESTAMP'] = current_df['TIMESTAMP'].dt.tz_localize(None)
    current_date_range_df = pd.DataFrame({
        'TIMESTAMP': pd.date_range(start=datetime(2009, 1, 1), end=current_df['TIMESTAMP'].max(), freq='D')
    }).sort_values('TIMESTAMP', ascending=True)
    current_df = pd.merge(current_df, current_date_range_df, on='TIMESTAMP', how='right')
    merged_df = pd.merge(merged_df, current_df, on='TIMESTAMP', how='outer').sort_values('TIMESTAMP').reset_index(drop=True)

new_map_path = "All_Crypto_Data/Macroeconomic_Data/Merged/Yahoo_Finance/1_Day/"
file_name = f"Yahoo_Finance_Daily_Macroeconomic_Data_USD_{merged_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"

if not os.path.exists(new_map_path):
    os.makedirs(new_map_path)

merged_df.to_csv(f"{new_map_path}{file_name}", index=False)

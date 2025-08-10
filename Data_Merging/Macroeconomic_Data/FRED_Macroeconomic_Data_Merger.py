import pandas as pd
import os
import re
from datetime import datetime

base_daily_path = "All_Crypto_Data/Macroeconomic_Data/Unmerged/FRED/1_Day/"
base_monthly_path = "All_Crypto_Data/Macroeconomic_Data/Unmerged/FRED/1_Month/"

merged_df = pd.DataFrame(columns={'TIMESTAMP':[]})

for f in os.listdir(base_daily_path):
    current_name = re.search(r'FRED_(.*?)_Daily', f).group(1).upper()
    current_df = pd.read_csv(f"{base_daily_path}{f}").rename(columns={'observation_date':'TIMESTAMP'})
    current_df = current_df.rename(columns={list(current_df.columns)[1]:current_name})
    current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])

    current_date_range_df = pd.DataFrame({
        'TIMESTAMP': pd.date_range(start=datetime(2009, 1, 1), end=current_df['TIMESTAMP'].max(), freq='D')
    }).sort_values('TIMESTAMP', ascending=True)

    current_df = pd.merge(current_df, current_date_range_df, on='TIMESTAMP', how='right')
    merged_df = pd.merge(merged_df, current_df, on='TIMESTAMP', how='outer').sort_values('TIMESTAMP').reset_index(drop=True)

for f in os.listdir(base_monthly_path):
    current_name = re.search(r'FRED_(.*?)_Monthly', f).group(1).upper()
    current_df = pd.read_csv(f"{base_monthly_path}{f}").rename(columns={'observation_date':'TIMESTAMP'})
    current_df = current_df.rename(columns={list(current_df.columns)[1]:current_name})
    current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])

    current_date_range_df = pd.DataFrame({
        'TIMESTAMP': pd.date_range(start=datetime(2009, 1, 1), end=current_df['TIMESTAMP'].max(), freq='D')
    }).sort_values('TIMESTAMP', ascending=True)

    current_df = pd.merge(current_df, current_date_range_df, on='TIMESTAMP', how='right')
    merged_df = pd.merge(merged_df, current_df, on='TIMESTAMP', how='outer').sort_values('TIMESTAMP').reset_index(drop=True)

new_map_path = "All_Crypto_Data/Macroeconomic_Data/Merged/FRED/1_Day/"
file_name = f"FRED_Daily_Macroeconomic_Data_USD_{merged_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"

if not os.path.exists(new_map_path):
    os.makedirs(new_map_path)

merged_df.to_csv(f"{new_map_path}{file_name}", index=False)

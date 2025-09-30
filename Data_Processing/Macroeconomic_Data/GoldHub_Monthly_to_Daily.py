import os
import pandas as pd
from datetime import datetime

# Script to generate GoldHub Gold Price dataset with daily intervals from the monthly GoldHub Gold Price dataset

prefix = 'GOLD_PRICE'

daily_file_path = "All_Crypto_Data/Macroeconomic_Data/Merged/GoldHub/1_Day/"
monthly_file_path = "All_Crypto_Data/Macroeconomic_Data/Merged/GoldHub/1_Month/"

monthly_gold_price_df = pd.read_excel(f"{monthly_file_path}GoldHub_Average_Monthly_Gold_Price_USD_Since_1978.xlsx").rename(columns={'Date':'TIMESTAMP'})
monthly_gold_price_df['TIMESTAMP'] = pd.to_datetime(monthly_gold_price_df['TIMESTAMP'])

new_column_names = {}

for c in monthly_gold_price_df.columns[1:]:
    new_column_names[c] = f"{prefix}_{c}"

monthly_gold_price_df = monthly_gold_price_df.rename(columns=new_column_names)

new_file_name_daily = f"GoldHub_Gold_Price_USD_Daily_{monthly_gold_price_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{monthly_gold_price_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"
new_file_name_monthly = f"GoldHub_Gold_Price_USD_Monthly_{monthly_gold_price_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{monthly_gold_price_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"

current_date_range_df = pd.DataFrame({
    'TIMESTAMP': pd.date_range(start=datetime(2009, 1, 1), end=monthly_gold_price_df['TIMESTAMP'].max(), freq='D')
    }).sort_values('TIMESTAMP', ascending=True).reset_index(drop=True)

daily_gold_price_df = pd.merge(monthly_gold_price_df, current_date_range_df, on='TIMESTAMP', how='right').sort_values('TIMESTAMP', ascending=True).reset_index(drop=True)

if not os.path.exists(f"{monthly_file_path}{new_file_name_monthly}"):
    if not os.path.exists(monthly_file_path):
        os.makedirs(monthly_file_path)
    
    monthly_gold_price_df.to_csv(f"{monthly_file_path}{new_file_name_monthly}", index=False)

if not os.path.exists(f"{daily_file_path}{new_file_name_daily}"):
    if not os.path.exists(daily_file_path):
        os.makedirs(daily_file_path)
    
    daily_gold_price_df.to_csv(f"{daily_file_path}{new_file_name_daily}", index=False)

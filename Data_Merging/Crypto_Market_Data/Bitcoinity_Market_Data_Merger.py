import pandas as pd
import os
import re

# Script to Merge Market Data into a single Dataset

base_file_path = "All_Crypto_Data/Crypto_Market_Data/Unmerged/Bitcoinity/BTC/1_Day/"
merged_df = pd.DataFrame({'TIMESTAMP':[]})

for csv in os.listdir(base_file_path):
    current_df = pd.read_csv(f"{base_file_path}{csv}").rename(columns={'Time':'TIMESTAMP'})
    current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])

    feature = re.search(r'UTC_(.*?)_(?=\d)', csv).group(1)

    if ('Bid_Ask_Spread' in feature) or ('Exchange_Rank' in feature) or ('Price_USD' in feature) or ('Trading_Volume' in feature):
        original_columns = current_df.columns[1:]
        new_columns = [f'BTC_{oc.upper()}_{feature.upper()}' for oc in original_columns]
    else:
        original_columns = current_df.columns[1:]
        new_columns = [f'BTC_{oc.upper()}' for oc in original_columns]
    
    current_df = current_df.rename(columns=dict(zip(original_columns, new_columns)))
    merged_df = pd.merge(merged_df, current_df, on='TIMESTAMP', how='outer').sort_values('TIMESTAMP').rename(columns={'BTC_PRICE_USD_y':'BTC_PRICE_USD'})

merged_df = merged_df.drop(labels=['BTC_PRICE_USD_x'], axis=1)

merged_file_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Bitcoinity/BTC/1_Day/"

if not os.path.exists(merged_file_path):
    os.makedirs(merged_file_path)

merged_df.to_csv(f"{merged_file_path}Bitcoinity_BTC_USD_Daily_UTC_Market_Data_{merged_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", index=False)

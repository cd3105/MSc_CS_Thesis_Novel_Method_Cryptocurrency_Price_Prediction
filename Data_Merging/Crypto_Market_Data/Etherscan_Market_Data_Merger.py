import pandas as pd
import os
import re

# Script to Merge Etherscan Market Data into a single Dataset

base_file_path = "All_Crypto_Data/Crypto_Market_Data/Unmerged/Etherscan/ETH/1_Day/"
merged_df = pd.DataFrame({'TIMESTAMP':[]})

for csv in os.listdir(base_file_path):
    current_df = pd.read_csv(f"{base_file_path}{csv}").rename(columns={'Date(UTC)':'TIMESTAMP'})
    current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])

    if 'UnixTimeStamp' in current_df.columns:
        current_df = current_df.drop(['UnixTimeStamp'], axis=1)
    
    if csv == 'Etherscan_ETH_USD_Daily_UTC_Market_Cap_USD_30_07_2015__14_05_2025.csv':
        current_df = current_df.drop(['Supply', 'Price_USD'], axis=1)
    
    feature = re.search(r'UTC_(.*?)_(?=\d)', csv).group(1)

    if (len(current_df.columns) == 2) and ('Value' in str(current_df.columns[1])):
        current_df = current_df.rename(columns={current_df.columns[1]:f'ETH_{feature.upper()}'})
    else:
        original_columns = current_df.columns[1:]
        new_columns = [f'ETH_{"_".join(oc.split(" ")).upper().replace("(", "").replace(")", "")}' for oc in original_columns]

        current_df = current_df.rename(columns=dict(zip(original_columns, new_columns)))
    
    merged_df = pd.merge(merged_df, current_df, on='TIMESTAMP', how='outer').sort_values('TIMESTAMP')

merged_df['ETH_MARKET_CAP_USD'] = merged_df['ETH_MARKET_CAP_USD'] * 10**6

merged_file_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Etherscan/ETH/1_Day/"

if not os.path.exists(merged_file_path):
    os.makedirs(merged_file_path)

merged_df.to_csv(f"{merged_file_path}Etherscan_ETH_USD_Daily_UTC_Market_Data_{merged_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", index=False)

import pandas as pd
import os
import re

base_file_path = "All_Crypto_Data/Blockchain_Data/Unmerged/Bitcoinity/BTC/1_Day/"
merged_df = pd.DataFrame({'TIMESTAMP':[]})

for csv in os.listdir(base_file_path):
    current_df = pd.read_csv(f"{base_file_path}{csv}").rename(columns={'Time':'TIMESTAMP'})
    current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])

    feature = re.search(r'UTC_(.*?)_(?=\d)', csv).group(1)

    if len(current_df.columns) == 2:
        current_df = current_df.rename(columns={'Unnamed: 1':f'BTC_{feature.upper()}'})
    else:
        original_columns = current_df.columns[1:]
        new_columns = [f'BTC_{feature.upper()}_{"_".join(oc.split(" ")).upper()}' for oc in original_columns]

        current_df = current_df.rename(columns=dict(zip(original_columns, new_columns)))
    
    merged_df = pd.merge(merged_df, current_df, on='TIMESTAMP', how='outer').sort_values('TIMESTAMP', ascending=True)

merged_file_path = "All_Crypto_Data/Blockchain_Data/Merged/Bitcoinity/BTC/1_Day/"

if not os.path.exists(merged_file_path):
    os.makedirs(merged_file_path)

merged_df.to_csv(f"{merged_file_path}Bitcoinity_BTC_Daily_UTC_Blockchain_Data_{merged_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", index=False)

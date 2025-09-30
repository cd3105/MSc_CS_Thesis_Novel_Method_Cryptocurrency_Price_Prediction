import pandas as pd
import os

# Script to Merge Google Trends Data into a single Dataset

ccs = ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']

base_path = 'All_Crypto_Data/Crypto_Google_Trends/Unmerged/Google_Trends/'

for cc in ccs:
    csvs = os.listdir(f"{base_path}{cc}/1_Month/")
    merged_df = pd.DataFrame({'TIMESTAMP':[]})

    for csv in csvs:
        current_df = pd.read_csv(f"{base_path}{cc}/1_Month/{csv}", skiprows=2).rename(columns={'Maand':'TIMESTAMP'})
        current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])
        current_df = current_df.sort_values('TIMESTAMP', ascending=True).rename(columns={current_df.columns[1]:csv[:-4]})

        current_date_range = pd.DataFrame({'TIMESTAMP':pd.date_range(current_df['TIMESTAMP'].min(), current_df['TIMESTAMP'].max())})
        current_df = pd.merge(current_df, current_date_range, how='right', on='TIMESTAMP')
        
        merged_df = pd.merge(merged_df, current_df, how='right', on='TIMESTAMP').sort_values('TIMESTAMP').reset_index(drop=True)
    
    merged_file_path = f'All_Crypto_Data/Crypto_Google_Trends/Merged/Google_Trends/{cc}/1_Day/'
    file_name = f"Google_Trends_{cc}_{merged_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"

    if not os.path.exists(merged_file_path):
        os.makedirs(merged_file_path)
    
    merged_df.to_csv(f"{merged_file_path}{file_name}", index=False)

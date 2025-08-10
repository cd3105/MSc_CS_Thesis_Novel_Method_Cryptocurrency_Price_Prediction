import pandas as pd
import os
import time

base_path = "All_Crypto_Data/Crypto_Market_Data/Unmerged/Blockchain/BTC/"
merged_daily_df = pd.DataFrame({'TIMESTAMP':[]})
merged_12_hourly_df = pd.DataFrame({'TIMESTAMP':[]})
merged_8_hourly_df = pd.DataFrame({'TIMESTAMP':[]})
merged_6_hourly_df = pd.DataFrame({'TIMESTAMP':[]})
merged_4_hourly_df = pd.DataFrame({'TIMESTAMP':[]})
merged_2_hourly_df = pd.DataFrame({'TIMESTAMP':[]})
merged_hourly_df = pd.DataFrame({'TIMESTAMP':[]})

for f in os.listdir(base_path):
    for csv in os.listdir(f'{base_path}{f}'):
        current_df = pd.read_csv(f'{base_path}{f}/{csv}')
        current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])
        current_df = current_df.set_index('TIMESTAMP')
        
        if f != '1_Day':
            current_daily_df = current_df.resample('D').agg({current_df.columns[0]: 'last',}).reset_index()
            current_12_hourly_df = current_df.resample('12H').agg({current_df.columns[0]: 'last',}).reset_index()
            current_8_hourly_df = current_df.resample('8H').agg({current_df.columns[0]: 'last',}).reset_index()
            current_6_hourly_df = current_df.resample('6H').agg({current_df.columns[0]: 'last',}).reset_index()
            current_4_hourly_df = current_df.resample('4H').agg({current_df.columns[0]: 'last',}).reset_index()
            current_2_hourly_df = current_df.resample('2H').agg({current_df.columns[0]: 'last',}).reset_index()
            current_hourly_df = current_df.resample('H').agg({current_df.columns[0]: 'last',}).reset_index()

            merged_12_hourly_df = pd.merge(merged_12_hourly_df, 
                                           current_12_hourly_df, 
                                           how='outer', 
                                           on='TIMESTAMP').sort_values('TIMESTAMP')

            merged_8_hourly_df = pd.merge(merged_8_hourly_df, 
                                          current_8_hourly_df, 
                                          how='outer', 
                                          on='TIMESTAMP').sort_values('TIMESTAMP')

            merged_6_hourly_df = pd.merge(merged_6_hourly_df, 
                                          current_6_hourly_df, 
                                          how='outer',
                                          on='TIMESTAMP').sort_values('TIMESTAMP')

            merged_4_hourly_df = pd.merge(merged_4_hourly_df,
                                          current_4_hourly_df, 
                                          how='outer', 
                                          on='TIMESTAMP').sort_values('TIMESTAMP')

            merged_2_hourly_df = pd.merge(merged_2_hourly_df, 
                                          current_2_hourly_df, 
                                          how='outer', 
                                          on='TIMESTAMP').sort_values('TIMESTAMP')

            merged_hourly_df = pd.merge(merged_hourly_df, 
                                        current_hourly_df, 
                                        how='outer', 
                                        on='TIMESTAMP').sort_values('TIMESTAMP')
        else:
            current_daily_df = current_df.reset_index()

        merged_daily_df = pd.merge(merged_daily_df, 
                                   current_daily_df, 
                                   how='outer', 
                                   on='TIMESTAMP').sort_values('TIMESTAMP')

new_daily_merged_file_path = 'All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain/BTC/1_Day/'
new_12_hourly_merged_file_path = 'All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain/BTC/12_Hour/'
new_8_hourly_merged_file_path = 'All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain/BTC/8_Hour/'
new_6_hourly_merged_file_path = 'All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain/BTC/6_Hour/'
new_4_hourly_merged_file_path = 'All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain/BTC/4_Hour/'
new_2_hourly_merged_file_path = 'All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain/BTC/2_Hour/'
new_hourly_merged_file_path = 'All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain/BTC/1_Hour/'

new_daily_merged_file_name = f"Blockchain_BTC_USD_Daily_Market_Data_{merged_daily_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_daily_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"
new_12_hourly_merged_file_name = f"Blockchain_BTC_USD_12_Hourly_Market_Data_{merged_12_hourly_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_12_hourly_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"
new_8_hourly_merged_file_name = f"Blockchain_BTC_USD_8_Hourly_Market_Data_{merged_8_hourly_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_8_hourly_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"
new_6_hourly_merged_file_name = f"Blockchain_BTC_USD_6_Hourly_Market_Data_{merged_6_hourly_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_6_hourly_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"
new_4_hourly_merged_file_name = f"Blockchain_BTC_USD_4_Hourly_Market_Data_{merged_4_hourly_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_4_hourly_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"
new_2_hourly_merged_file_name = f"Blockchain_BTC_USD_2_Hourly_Market_Data_{merged_2_hourly_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_2_hourly_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"
new_hourly_merged_file_name = f"Blockchain_BTC_USD_Hourly_Market_Data_{merged_hourly_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_hourly_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"

if not os.path.exists(new_daily_merged_file_path):
    os.makedirs(new_daily_merged_file_path)

if not os.path.exists(new_12_hourly_merged_file_path):
    os.makedirs(new_12_hourly_merged_file_path)

if not os.path.exists(new_8_hourly_merged_file_path):
    os.makedirs(new_8_hourly_merged_file_path)

if not os.path.exists(new_6_hourly_merged_file_path):
    os.makedirs(new_6_hourly_merged_file_path)

if not os.path.exists(new_4_hourly_merged_file_path):
    os.makedirs(new_4_hourly_merged_file_path)

if not os.path.exists(new_2_hourly_merged_file_path):
    os.makedirs(new_2_hourly_merged_file_path)

if not os.path.exists(new_hourly_merged_file_path):
    os.makedirs(new_hourly_merged_file_path)

merged_daily_df.to_csv(f'{new_daily_merged_file_path}{new_daily_merged_file_name}', 
                       index=False)
merged_12_hourly_df.to_csv(f'{new_12_hourly_merged_file_path}{new_12_hourly_merged_file_name}', 
                           index=False)
merged_8_hourly_df.to_csv(f'{new_8_hourly_merged_file_path}{new_8_hourly_merged_file_name}', 
                          index=False)
merged_6_hourly_df.to_csv(f'{new_6_hourly_merged_file_path}{new_6_hourly_merged_file_name}', 
                          index=False)
merged_4_hourly_df.to_csv(f'{new_4_hourly_merged_file_path}{new_4_hourly_merged_file_name}', 
                          index=False)
merged_2_hourly_df.to_csv(f'{new_2_hourly_merged_file_path}{new_2_hourly_merged_file_name}', 
                          index=False)
merged_hourly_df.to_csv(f'{new_hourly_merged_file_path}{new_hourly_merged_file_name}', 
                        index=False)

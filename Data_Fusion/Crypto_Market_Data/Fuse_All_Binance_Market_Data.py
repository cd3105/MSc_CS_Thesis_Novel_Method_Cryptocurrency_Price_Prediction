import pandas as pd
import os

# Script for Fusing Binance Market Data

original_data_base_path = "All_Crypto_Data/Original_Data/"
new_data_base_path_1 = "All_Crypto_Data/Crypto_Market_Data/Merged/Binance/" # Will be ignored after evaluation of data
new_data_base_path_2 = "All_Crypto_Data/Crypto_Market_Data/Merged/Binance_via_CoinDesk/" 
new_data_base_path_3 = "All_Crypto_Data/Crypto_Market_Data/Merged/Binance_via_CryptoDataDownload/" # Will be ignored after evaluation of data
new_data_base_path_4 = "All_Crypto_Data/Crypto_Market_Data/Merged/Binance_via_Investing/" # Will be ignored after evaluation of data

ccs = ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']

for cc in ccs:
    original_data_file_name = os.listdir(f'{original_data_base_path}{cc}/1_Day/')[0]
    new_data_file_name_2 = os.listdir(f'{new_data_base_path_2}{cc}/1_Day/')[0]

    original_data_df = pd.read_csv(f'{original_data_base_path}{cc}/1_Day/{original_data_file_name}').rename(columns={f'{cc}_OPENING_PRICE_USD':f'{cc}_OPEN_PRICE_USD'})
    new_data_df_2 = pd.read_csv(f'{new_data_base_path_2}{cc}/1_Day/{new_data_file_name_2}')

    original_data_df['TIMESTAMP'] = pd.to_datetime(original_data_df['TIMESTAMP'])
    new_data_df_2['TIMESTAMP'] = pd.to_datetime(new_data_df_2['TIMESTAMP'])

    # cns_to_adjust = [c for c in new_data_df_2.columns if 'USDT' in c]
    # adjusted_cns = [c.replace('USDT', 'USD') for c in new_data_df_2.columns if 'USDT' in c]
    # new_data_df_2 = new_data_df_2.rename(columns=dict(zip(cns_to_adjust, adjusted_cns)))
    # combined_df = pd.merge(original_data_df, new_data_df_2, on='TIMESTAMP', how='outer')

    cns_to_adjust = [c for c in original_data_df.columns if 'USD' in c]
    adjusted_cns = [c.replace('USD', 'USDT') for c in original_data_df.columns if 'USD' in c]
    original_data_df = original_data_df.rename(columns=dict(zip(cns_to_adjust, adjusted_cns)))
    combined_df = pd.merge(original_data_df, new_data_df_2, on='TIMESTAMP', how='outer')
    
    columns_to_reorder = []

    for c in [c for c in combined_df.columns if '_x' in c]:
        current_column = c[:-2]
        combined_df[current_column] = combined_df[f'{current_column}_x'].combine_first(combined_df[f'{current_column}_y'])
        combined_df = combined_df.drop([f'{current_column}_x', f'{current_column}_y'], axis=1)

        columns_to_reorder.append(current_column)

    columns_reordered = ['TIMESTAMP'] + columns_to_reorder + [c for c in combined_df.columns if c not in (['TIMESTAMP'] + columns_to_reorder)]
    combined_df = combined_df[columns_reordered]
    
    save_path = f'All_Crypto_Data/Crypto_Market_Data/Merged/Binance_Extended/{cc}/1_Day/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    combined_df.to_csv(f"{save_path}Binance_{cc}_USDT_Daily_Market_Data_{combined_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{combined_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                       index=False)

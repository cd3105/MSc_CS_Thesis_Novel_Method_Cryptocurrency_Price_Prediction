import pandas as pd
import os
import zipfile
from datetime import datetime

# Script for unzipping Binance Data and concatenating the monthly data

base_path = "All_Crypto_Data/Crypto_Market_Data/Unmerged/Binance/"
frequencies = ['1_Day', '1_Hour']

def detect_timestamp_unit(ts):
    if len(str(ts)) == 13:
        return 1000
    elif len(str(ts)) == 16:
        return 1000000
    elif len(str(ts)) == 19:
        return 1000000000
    else:
        return 1

for cc in [m for m in os.listdir(base_path) if not m.endswith('.txt')]:
    for f in frequencies:
        current_cc_path = f"{base_path}{cc}/{f}/"
        current_concat_df = pd.DataFrame(columns=['TIMESTAMP',
                                                  f'{cc}_OPENING_PRICE_USDT',
                                                  f'{cc}_HIGH_PRICE_USDT',
                                                  f'{cc}_LOW_PRICE_USDT',
                                                  f'{cc}_CLOSE_PRICE_USDT',
                                                  f'{cc}_VOLUME_USDT',
                                                  'CLOSE_TIMESTAMP',
                                                  f'{cc}_QUOTE_ASSET_VOLUME_USDT',
                                                  f'{cc}_TRADE_COUNT',
                                                  f'{cc}_TAKER_BUY_BASE_ASSET_VOLUME',
                                                  f'{cc}_TAKER_BUY_QUOTE_ASSET_VOLUME',
                                                  'IGNORE',])

        for zf in [zf for zf in os.listdir(current_cc_path) if zf.endswith('.zip')]:
            zip_file_path = os.path.join(current_cc_path, zf)

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(current_cc_path)
        
        for csv in [csv for csv in os.listdir(current_cc_path) if csv.endswith('.csv')]:
            csv_file_path = os.path.join(current_cc_path, csv)

            current_df = pd.read_csv(csv_file_path, names=current_concat_df.columns)
            current_concat_df = pd.concat([current_concat_df, current_df]).reset_index(drop=True).sort_values('TIMESTAMP', ascending=True)
        
        current_concat_df['TIMESTAMP'] = current_concat_df['TIMESTAMP'].apply(lambda x: datetime.utcfromtimestamp(int(x / detect_timestamp_unit(x))))
        current_concat_df = current_concat_df.drop(['CLOSE_TIMESTAMP', 'IGNORE'], axis=1)

        if f == '1_Day':
            full_range = pd.date_range(start=current_concat_df['TIMESTAMP'].min(), end=current_concat_df['TIMESTAMP'].max(), freq='D')
        else:
            full_range = pd.date_range(start=current_concat_df['TIMESTAMP'].min(), end=current_concat_df['TIMESTAMP'].max(), freq='H')
        
        current_concat_df = pd.merge(current_concat_df, pd.DataFrame({'TIMESTAMP':full_range}), how='right', on='TIMESTAMP')
        
        new_concat_file_name = f"Binance_{cc}_USDT_Daily_Market_Data_{current_concat_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_concat_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"
        new_concat_map_path = f'All_Crypto_Data/Crypto_Market_Data/Merged/Binance/{cc}/{f}/'

        if not os.path.exists(new_concat_map_path):
            os.makedirs(new_concat_map_path)

        current_concat_df.to_csv(f'{new_concat_map_path}{new_concat_file_name}', index=False)

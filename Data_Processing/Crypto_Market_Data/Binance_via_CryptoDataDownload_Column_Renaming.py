import pandas as pd
import os
from datetime import datetime

# Script for renaming columns corresponding to Binance Data retrieved via CryptoDataDownload

def detect_timestamp_unit(ts):
    if len(str(ts)) == 13:
        return 1000
    elif len(str(ts)) == 16:
        return 1000000
    elif len(str(ts)) == 19:
        return 1000000000
    else:
        return 1
    
base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Binance_via_CryptoDataDownload/"

for cc in [m for m in os.listdir(base_path) if not m.endswith('txt')]:
    for f in os.listdir(f'{base_path}{cc}'):
        current_cc_path = f'{base_path}{cc}/{f}/'
        current_cc_fn = os.listdir(current_cc_path)[0]
        current_cc_df = pd.read_csv(f'{current_cc_path}{current_cc_fn}', skiprows=1).drop(['Date', 'Symbol'], axis=1).rename(columns={'Unix':'TIMESTAMP',
                                                                                                                                      'Open':f'{cc}_OPEN_PRICE_USDT',
                                                                                                                                      'High':f'{cc}_HIGH_PRICE_USDT',
                                                                                                                                      'Low':f'{cc}_LOW_PRICE_USDT',
                                                                                                                                      'Close':f'{cc}_CLOSE_PRICE_USDT',
                                                                                                                                      f'Volume {cc}':f'{cc}_VOLUME_USDT',
                                                                                                                                      f'Volume USDT':f'VOLUME_USDT',
                                                                                                                                      'tradecount':f'{cc}_TRADE_COUNT',})
        current_cc_df['TIMESTAMP'] = current_cc_df['TIMESTAMP'].apply(lambda x: datetime.utcfromtimestamp(x / detect_timestamp_unit(x)))
        current_cc_df = current_cc_df.sort_values('TIMESTAMP').reset_index(drop=True)
        current_cc_df.to_csv(f'{current_cc_path}{current_cc_fn}', index=False)

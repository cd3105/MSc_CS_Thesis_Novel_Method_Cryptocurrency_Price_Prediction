import pandas as pd
import os
from datetime import datetime

# Script for renaming Binance Data retrieved via the Coindesk API

base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Binance_via_CoinDesk/"

for cc in [m for m in os.listdir(base_path) if not m.endswith('.txt')]:
    for f in os.listdir(f'{base_path}{cc}/'):
        current_cc_path = f'{base_path}{cc}/{f}/'
        current_cc_csv = os.listdir(current_cc_path)[0]
        current_cc_df = pd.read_csv(f'{current_cc_path}{current_cc_csv}', index_col=0).drop(['UNIT',
                                                                                             'TYPE',
                                                                                             'MARKET',
                                                                                             'INSTRUMENT',
                                                                                             'MAPPED_INSTRUMENT',
                                                                                             'BASE', 
                                                                                             'QUOTE',
                                                                                             'BASE_ID', 
                                                                                             'QUOTE_ID',
                                                                                             'TRANSFORM_FUNCTION',
                                                                                             'FIRST_TRADE_TIMESTAMP', 
                                                                                             'LAST_TRADE_TIMESTAMP',
                                                                                             'HIGH_TRADE_TIMESTAMP',
                                                                                             'LOW_TRADE_TIMESTAMP',
                                                                                             'QUOTE_VOLUME_UNKNOWN',
                                                                                             'TOTAL_TRADES_UNKNOWN',
                                                                                             'VOLUME_UNKNOWN'], axis=1).rename(columns={'OPEN':f'{cc}_OPEN_PRICE_USDT',
                                                                                                                                        'CLOSE':f'{cc}_CLOSE_PRICE_USDT',
                                                                                                                                        'HIGH':f'{cc}_HIGH_PRICE_USDT',
                                                                                                                                        'LOW':f'{cc}_LOW_PRICE_USDT',
                                                                                                                                        'FIRST_TRADE_PRICE':f'{cc}_FIRST_TRADE_PRICE_USDT',
                                                                                                                                        'HIGH_TRADE_PRICE':f'{cc}_HIGH_TRADE_PRICE_USDT',
                                                                                                                                        'LOW_TRADE_PRICE':f'{cc}_LOW_TRADE_PRICE_USDT',
                                                                                                                                        'LAST_TRADE_PRICE':f'{cc}_LAST_TRADE_PRICE_USDT',
                                                                                                                                        'VOLUME':f'{cc}_VOLUME_USDT',
                                                                                                                                        'TOTAL_TRADES':f'{cc}_TOTAL_TRADE_COUNT',
                                                                                                                                        'TOTAL_TRADES_BUY':f'{cc}_TOTAL_TRADES_BUY',
                                                                                                                                        'TOTAL_TRADES_SELL':f'{cc}_TOTAL_TRADES_SELL',
                                                                                                                                        'QUOTE_VOLUME':f'{cc}_QUOTE_VOLUME',
                                                                                                                                        'VOLUME_BUY':f'{cc}_VOLUME_BUY',
                                                                                                                                        'QUOTE_VOLUME_BUY':f'{cc}_QUOTE_VOLUME_BUY',
                                                                                                                                        'VOLUME_SELL':f'{cc}_VOLUME_SELL',
                                                                                                                                        'QUOTE_VOLUME_SELL':f'{cc}_QUOTE_VOLUME_SELL',})
        current_cc_df['TIMESTAMP'] = current_cc_df['TIMESTAMP'].apply(lambda x: datetime.utcfromtimestamp(x))
        current_cc_df.sort_values('TIMESTAMP').reset_index(drop=True).to_csv(f'{current_cc_path}{current_cc_csv}', index=False)

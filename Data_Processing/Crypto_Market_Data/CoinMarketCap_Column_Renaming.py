import pandas as pd
import os

# Script for renaming columns in CoinMarketCap data

base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/CoinMarketCap/"

for cc in [f for f in os.listdir(base_path) if not f.endswith('.txt')]:
    current_file_name = os.listdir(f"{base_path}{cc}/1_Day/")[0]
    current_cc_df = pd.read_csv(f"{base_path}{cc}/1_Day/{current_file_name}", delimiter=';').drop(['timeClose', 'timeHigh', 'timeLow', 'name', 'timestamp'], axis=1).rename(columns={'timeOpen':'TIMESTAMP',
                                                                                                                                                                                     'open':f'{cc}_OPEN_PRICE_USD',
                                                                                                                                                                                     'high':f'{cc}_HIGH_PRICE_USD',
                                                                                                                                                                                     'low':f'{cc}_LOW_PRICE_USD',
                                                                                                                                                                                     'close':f'{cc}_CLOSE_PRICE_USD',
                                                                                                                                                                                     'volume':f'{cc}_VOLUME_USD',
                                                                                                                                                                                     'marketCap':f'{cc}_MARKET_CAP_USD',})
    current_cc_df['TIMESTAMP'] = pd.to_datetime(current_cc_df['TIMESTAMP'])
    current_cc_df[f'{cc}_OPEN_PRICE_USD'] = current_cc_df[f'{cc}_OPEN_PRICE_USD'].apply(lambda x: float(x))
    current_cc_df[f'{cc}_HIGH_PRICE_USD'] = current_cc_df[f'{cc}_HIGH_PRICE_USD'].apply(lambda x: float(x))
    current_cc_df[f'{cc}_LOW_PRICE_USD'] = current_cc_df[f'{cc}_LOW_PRICE_USD'].apply(lambda x: float(x))
    current_cc_df[f'{cc}_CLOSE_PRICE_USD'] = current_cc_df[f'{cc}_CLOSE_PRICE_USD'].apply(lambda x: float(x))
    current_cc_df[f'{cc}_VOLUME_USD'] = current_cc_df[f'{cc}_VOLUME_USD'].apply(lambda x: float(x))
    current_cc_df[f'{cc}_MARKET_CAP_USD'] = current_cc_df[f'{cc}_MARKET_CAP_USD'].apply(lambda x: float(x))
    
    current_cc_df.sort_values('TIMESTAMP', ascending=True).reset_index(drop=True).to_csv(f"{base_path}{cc}/1_Day/{current_file_name}", index=False)

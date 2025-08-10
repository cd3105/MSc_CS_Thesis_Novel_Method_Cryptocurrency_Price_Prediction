import pandas as pd
import datetime
import os

map_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Kaggle/Kaggle_Daily_USD_Market_Data_23_CCs_until_07_07_2021/"

for f in [f for f in os.listdir(map_path) if f.endswith(".csv")]:
    file_path = f"{map_path}{f}"
    current_cc_df = pd.read_csv(file_path)
    current_cc = current_cc_df.iloc[0]['Symbol']
    current_cc_df = current_cc_df.drop(["Symbol", "SNo", "Name"], axis=1).rename(columns={'Date':'TIMESTAMP',
                                                                                          'Open':f'{current_cc}_OPEN_PRICE_USD',
                                                                                          'High':f'{current_cc}_HIGH_PRICE_USD',
                                                                                          'Low':f'{current_cc}_LOW_PRICE_USD',
                                                                                          'Close':f'{current_cc}_CLOSE_PRICE_USD',
                                                                                          'Volume':f'{current_cc}_VOLUME_USD',
                                                                                          'Marketcap':f'{current_cc}_MARKET_CAP_USD'})

    current_cc_df['TIMESTAMP'] = pd.to_datetime(current_cc_df['TIMESTAMP'])
    current_cc_df = current_cc_df.sort_values(by='TIMESTAMP', ascending=True)
    new_file_name = f"Kaggle_{current_cc}_Daily_USD_OHLCV_{current_cc_df.iloc[0]['TIMESTAMP'].strftime('%d_%m_%Y')}__{current_cc_df.iloc[-1]['TIMESTAMP'].strftime('%d_%m_%Y')}.csv"
    new_directory_path = f"{map_path}{current_cc}/"
    
    if not os.path.exists(new_directory_path):
        os.mkdir(new_directory_path)
    
    current_cc_df.to_csv(f"{new_directory_path}{new_file_name}", index=False)

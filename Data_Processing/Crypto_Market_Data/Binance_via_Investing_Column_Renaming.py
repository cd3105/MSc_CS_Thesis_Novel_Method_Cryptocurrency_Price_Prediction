import pandas as pd
import os

# Script for renaming columns in Investing data

base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Binance_via_Investing/"

for cc in [dir for dir in os.listdir(base_path) if not dir.endswith('.txt')]:
    current_cc_path = f'{base_path}/{cc}/1_Day/'
    current_cc_csv = os.listdir(current_cc_path)[0]
    current_cc_df = pd.read_csv(f'{current_cc_path}/{current_cc_csv}').drop(columns=['Vol.',
                                                                                     'Change %'])
    
    current_cc_df['Date'] = pd.to_datetime(current_cc_df['Date'])
    current_cc_df['Price'] = current_cc_df['Price'].apply(lambda x: float(str(x).replace(',', '')))
    current_cc_df['Open'] = current_cc_df['Open'].apply(lambda x: float(str(x).replace(',', '')))
    current_cc_df['High'] = current_cc_df['High'].apply(lambda x: float(str(x).replace(',', '')))
    current_cc_df['Low'] = current_cc_df['Low'].apply(lambda x: float(str(x).replace(',', '')))

    current_cc_df = current_cc_df.sort_values('Date').reset_index(drop=True).rename(columns={'Date':'TIMESTAMP',
                                                                                             'Price':f'{cc}_CLOSE_PRICE_USD',
                                                                                             'Open':f'{cc}_OPEN_PRICE_USD',
                                                                                             'High':f'{cc}_HIGH_PRICE_USD',
                                                                                             'Low':f'{cc}_LOW_PRICE_USD',})
    
    current_cc_df = current_cc_df.sort_values('TIMESTAMP').reset_index(drop=True)
    current_cc_df.to_csv(f'{current_cc_path}/{current_cc_csv}', 
                         index=False)

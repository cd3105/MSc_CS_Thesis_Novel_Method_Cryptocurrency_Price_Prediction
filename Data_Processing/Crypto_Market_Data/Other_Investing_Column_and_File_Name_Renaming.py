import pandas as pd
import os

# Script for renaming columns in Investing data

other_investing_data_base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Other_Investing/"

for cc_csv in [m for m in os.listdir(other_investing_data_base_path) if m.endswith('.csv')]:
    current_cc = cc_csv.split('_')[0]

    current_cc_df = pd.read_csv(f"{other_investing_data_base_path}{cc_csv}")
    current_cc_df = current_cc_df.drop(columns=['Vol.',
                                                'Change %'])

    current_cc_df['Date'] = pd.to_datetime(current_cc_df['Date'])
    current_cc_df['Price'] = current_cc_df['Price'].apply(lambda x: float(str(x).replace(',', '')))
    current_cc_df['Open'] = current_cc_df['Open'].apply(lambda x: float(str(x).replace(',', '')))
    current_cc_df['High'] = current_cc_df['High'].apply(lambda x: float(str(x).replace(',', '')))
    current_cc_df['Low'] = current_cc_df['Low'].apply(lambda x: float(str(x).replace(',', '')))
        
    current_cc_df = current_cc_df.sort_values('Date').reset_index(drop=True).rename(columns={'Date':'TIMESTAMP',
                                                                                             'Price':f'{current_cc}_CLOSE_PRICE_USD',
                                                                                             'Open':f'{current_cc}_OPEN_PRICE_USD',
                                                                                             'High':f'{current_cc}_HIGH_PRICE_USD',
                                                                                             'Low':f'{current_cc}_LOW_PRICE_USD',})
    
    current_cc_path = f"{other_investing_data_base_path}{current_cc}/1_Day/"

    if not os.path.exists(current_cc_path):
        os.makedirs(current_cc_path)

    current_cc_df.to_csv(f"{current_cc_path}Investing_{current_cc}_USD_Daily_OHLC_Data_{current_cc_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_cc_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                         index=False)
    
    os.remove(f"{other_investing_data_base_path}{cc_csv}")

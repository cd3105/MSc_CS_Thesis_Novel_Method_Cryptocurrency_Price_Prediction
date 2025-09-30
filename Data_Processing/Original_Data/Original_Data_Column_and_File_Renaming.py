import pandas as pd
import os
from datetime import datetime

# Script for renaming file and columns of Original Data

original_data_path = "Original_Code/saved_data/"

for f in os.listdir(original_data_path):
    current_cc = f.replace('.csv', '').split('_')[-1].upper()
    current_cc_df = pd.read_csv(f'{original_data_path}{f}', index_col=0).rename(columns={'date':'TIMESTAMP',
                                                                                         'open':f'{current_cc}_OPENING_PRICE_USD',
                                                                                         'high':f'{current_cc}_HIGH_PRICE_USD',
                                                                                         'low':f'{current_cc}_LOW_PRICE_USD',
                                                                                         'close':f'{current_cc}_CLOSE_PRICE_USD',
                                                                                         'timestamp':'ts'})
    current_cc_df['TIMESTAMP'] = current_cc_df['ts'].apply(lambda x: datetime.utcfromtimestamp(x))
    current_cc_df = current_cc_df.drop(['ts'], axis=1).sort_values('TIMESTAMP').reset_index(drop=True)

    new_cc_df_file_name = f"Binance_and_Investing_{current_cc}_Daily_Market_Data_{current_cc_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_cc_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"
    new_cc_df_file_path = f'All_Crypto_Data/Original_Data/{current_cc}/1_Day/'
    
    if not os.path.exists(new_cc_df_file_path):
        os.makedirs(new_cc_df_file_path)
    
    current_cc_df.to_csv(f'{new_cc_df_file_path}{new_cc_df_file_name}', index=False)

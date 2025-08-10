import pandas as pd
import os
from datetime import datetime, timedelta

unprocessed_other_data_base_path = "All_Crypto_Data/New_Data/Unprocessed/Other_Social_Data/"
selected_ccs = ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']
selected_start_date = datetime(2017, 6, 1)
selected_end_date = datetime(2024, 1, 1)
selected_date_range = pd.date_range(selected_start_date,
                                    selected_end_date,
                                    freq='D')
selected_date_range_df = pd.DataFrame({'TIMESTAMP':selected_date_range})

for cc in selected_ccs:
    current_other_data_path = f"{unprocessed_other_data_base_path}All_Other_Social_Data/{cc}/1_Day/"
    current_other_data_file = os.listdir(current_other_data_path)[0]
    current_other_data_df = pd.read_csv(f"{current_other_data_path}{current_other_data_file}")
    current_other_data_df['TIMESTAMP'] = pd.to_datetime(current_other_data_df['TIMESTAMP'])
    columns_to_drop = []

    current_processed_other_data_df = pd.merge(selected_date_range_df,
                                               current_other_data_df,
                                               on="TIMESTAMP",
                                               how="left")
    
    for column in current_processed_other_data_df.columns[1:]:
        current_column_df = current_processed_other_data_df[['TIMESTAMP', column]]
        current_min_timestamp = current_column_df.dropna()['TIMESTAMP'].min()
        current_max_timestamp = current_column_df.dropna()['TIMESTAMP'].max()
        current_column_df = current_column_df[(current_column_df['TIMESTAMP'] >= current_min_timestamp) & (current_column_df['TIMESTAMP'] <= current_max_timestamp)]

        print(f"Current Column: {column}")
        print(f"\t- First Date Available: {current_min_timestamp}")
        print(f"\t- Last Date Available: {current_max_timestamp}")
        print(f"\t- NaN Check: {current_column_df[current_column_df[column].isna()]}\n")

    if cc != "XRP": # No XRP Other Data Available in Selected Period
        current_processed_other_data_df = current_processed_other_data_df[['TIMESTAMP', 
                                                                           f'{cc}_GOOGLE_TRENDS_FULL_NAME_PER_BIC']] # No Tweet Volume Data Available in Selected Period
        
        current_processed_other_data_base_path = f"All_Crypto_Data/New_Data/Processed/Other_Social_Data/{cc}/1_Day/"

        if not os.path.exists(current_processed_other_data_base_path):
            os.makedirs(current_processed_other_data_base_path)

        current_processed_other_data_df.to_csv(f"{current_processed_other_data_base_path}All_Other_Social_Data_{cc}_USD_Daily_{current_processed_other_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_processed_other_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                               index=False)


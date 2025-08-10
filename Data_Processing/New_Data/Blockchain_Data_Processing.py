import pandas as pd
import os
from datetime import datetime, timedelta

unprocessed_all_blockchain_data_base_path = "All_Crypto_Data/New_Data/Unprocessed/Blockchain_Data/All_Blockchain_Data/"
unprocessed_only_bic_data_base_path = "All_Crypto_Data/New_Data/Unprocessed/Blockchain_Data/Only_BitInfoCharts_Blockchain_Data/"

selected_ccs = ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']
selected_start_date = datetime(2017, 6, 1)
selected_end_date = datetime(2024, 12, 31)
selected_date_range = pd.date_range(selected_start_date,
                                    selected_end_date,
                                    freq='D')
selected_date_range_df = pd.DataFrame({'TIMESTAMP':selected_date_range})


for cc in os.listdir(unprocessed_only_bic_data_base_path):
    current_only_bic_data_path = f"{unprocessed_only_bic_data_base_path}{cc}/1_Day/"
    current_only_bic_data_file = os.listdir(current_only_bic_data_path)[0]
    current_only_bic_data_df = pd.read_csv(f"{current_only_bic_data_path}{current_only_bic_data_file}")
    current_only_bic_data_df['TIMESTAMP'] = pd.to_datetime(current_only_bic_data_df['TIMESTAMP'])
    
    current_only_bic_data_df = current_only_bic_data_df.dropna(axis=1, how='all')
    current_only_bic_data_df = pd.merge(selected_date_range_df, 
                                        current_only_bic_data_df,
                                        on='TIMESTAMP',
                                        how='left')
    print(current_only_bic_data_df[current_only_bic_data_df.isna().any(axis=1)])


# for cc in selected_ccs:
#     current_blockchain_data_path = f"{unprocessed_blockchain_data_base_path}All_Blockchain_Data/{cc}/1_Day/"
#     current_blockchain_data_file = os.listdir(current_blockchain_data_path)[0]
#     current_blockchain_data_df = pd.read_csv(f"{current_blockchain_data_path}{current_blockchain_data_file}")
#     current_blockchain_data_df['TIMESTAMP'] = pd.to_datetime(current_blockchain_data_df['TIMESTAMP'])
#     columns_to_drop = []

#     current_processed_blockchain_data_df = pd.merge(selected_date_range_df,
#                                                     current_blockchain_data_df,
#                                                     on="TIMESTAMP",
#                                                     how="left")
    
#     for column in current_processed_blockchain_data_df.columns[1:]:
#         current_column_df = current_processed_blockchain_data_df[['TIMESTAMP', column]]
#         current_min_timestamp = current_column_df.dropna()['TIMESTAMP'].min()
#         current_max_timestamp = current_column_df.dropna()['TIMESTAMP'].max()
#         current_column_df = current_column_df[(current_column_df['TIMESTAMP'] >= current_min_timestamp) & (current_column_df['TIMESTAMP'] <= current_max_timestamp)]

#         print(f"Current Column: {column}")
#         print(f"\t- First Date Available: {current_min_timestamp}")
#         print(f"\t- Last Date Available: {current_max_timestamp}")
#         print(f"\t- NaN Check: {current_column_df[current_column_df[column].isna()]}\n")

#     if cc != "XRP": # No XRP Blockchain Data Available in Selected Period
#         if cc == "BTC":
#             current_processed_blockchain_data_df = current_processed_blockchain_data_df.drop([f"{cc}_CUMULATIVE_DAYS_DESTROYED_PER_BLOCKCHAIN", 
#                                                                                               f"{cc}_DAYS_DESTROYED_PER_BLOCKCHAIN", 
#                                                                                               f"{cc}_MINIMUM_AGE_1_WEEK_DAYS_DESTROYED_PER_BLOCKCHAIN", 
#                                                                                               f"{cc}_MINIMUM_AGE_1_MONTH_DAYS_DESTROYED_PER_BLOCKCHAIN",
#                                                                                               f"{cc}_MINIMUM_AGE_1_YEAR_DAYS_DESTROYED_PER_BLOCKCHAIN", 
#                                                                                               f"{cc}_NETWORK_DEFICIT_PER_BLOCKCHAIN", 
#                                                                                               f"{cc}_TRADE_VOLUME_VS_TRANSACTION_VOLUME_RATIO_PER_BLOCKCHAIN"], 
#                                                                                               axis=1)

#         if cc == "ETH":
#             current_processed_blockchain_data_df = current_processed_blockchain_data_df.drop([f"{cc}_MEDIAN_TRANSACTION_VALUE_USD_PER_BIC", 
#                                                                                               f"{cc}_MINING_PROFITABILITY_PER_BIC"], 
#                                                                                              axis=1)

#         if cc == "XMR":
#             current_processed_blockchain_data_df = current_processed_blockchain_data_df.drop([f"{cc}_ACTIVE_ADDRESS_COUNT_PER_BIC", 
#                                                                                               f"{cc}_AVERAGE_TRANSACTION_VALUE_USD_PER_BIC", 
#                                                                                               f"{cc}_MEDIAN_TRANSACTION_FEE_USD_PER_BIC", 
#                                                                                               f"{cc}_MEDIAN_TRANSACTION_VALUE_USD_PER_BIC", 
#                                                                                               f"{cc}_SENT_COINS_USD_PER_BIC", 
#                                                                                               f"{cc}_SENT_FROM_UNIQUE_ADDRESS_COUNT_PER_BIC",
#                                                                                               f"{cc}_TOP_100_RICHEST_PERCENTAGE_OF_TOTAL_COINS_PER_BIC"], 
#                                                                                               axis=1)
    
        #print(f"NaN Check: {current_processed_blockchain_data_df[current_processed_blockchain_data_df.isna().any(axis=1)]}")
        
    #     current_processed_other_data_base_path = f"All_Crypto_Data/New_Data/Processed/Other_Data/{cc}/1_Day/"

    #     if not os.path.exists(current_processed_other_data_base_path):
    #         os.makedirs(current_processed_other_data_base_path)

    #     current_processed_other_data_df.to_csv(f"{current_processed_other_data_base_path}All_Other_Data_{cc}_USD_Daily_{current_processed_other_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_processed_other_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
    #                                            index=False)


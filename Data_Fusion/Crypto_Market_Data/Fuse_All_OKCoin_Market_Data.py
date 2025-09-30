import pandas as pd
import os
from datetime import datetime

# Script for Fusing OKCoin Market Data

base_file_path_1 = "All_Crypto_Data/Crypto_Market_Data/OKCoin_via_CoinDesk"
base_file_path_2 = "All_Crypto_Data/Crypto_Market_Data/OKCoin_via_CryptoCompare"
base_file_path_3 = "All_Crypto_Data/Crypto_Market_Data/OKCoin_via_CryptoDataDownload"

frequencies = ['1_Day', '1_Hour']

for cc in [m for m in os.listdir(base_file_path_1) if not m.endswith('.txt')][:1]:
    for f in frequencies[:1]:
        file_name_1 = os.listdir(f"{base_file_path_1}/{cc}/{f}/")[0]

        current_df_1 = pd.read_csv(f"{base_file_path_1}/{cc}/{f}/{file_name_1}", 
                                   index_col=0).drop(['UNIT',
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
                                                      'VOLUME_UNKNOWN'], axis=1).rename(columns={'OPEN':'OPEN_PRICE',
                                                                                                 'CLOSE':'CLOSE_PRICE',
                                                                                                 'HIGH':'HIGH_PRICE',
                                                                                                 'LOW':'LOW_PRICE',
                                                                                                 'VOLUME':'VOLUME_USD',
                                                                                                 'TOTAL_TRADES':'TOTAL_TRADE_COUNT'}).sort_values('TIMESTAMP', 
                                                                                                                                                  ascending=True).reset_index(drop=True)
        current_df_1['DATE'] = current_df_1['TIMESTAMP'].apply(lambda x: datetime.utcfromtimestamp(x))

        print(current_df_1[current_df_1['TIMESTAMP'] == 1485216000])

        file_name_3 = os.listdir(f"{base_file_path_3}/{cc}/{f}/")[0]
        current_df_3 = pd.read_csv(f"{base_file_path_3}/{cc}/{f}/{file_name_3}").drop(['Symbol'], axis=1).rename(columns={'Unix Timestamp':'TIMESTAMP',
                                                                                                                           'Date':'DATE', 
                                                                                                                           'Open':'OPEN_PRICE',
                                                                                                                           'High':'HIGH_PRICE',
                                                                                                                           'Low':'LOW_PRICE',
                                                                                                                           'Close':'CLOSE_PRICE',
                                                                                                                           'Volume BTC':'VOLUME_BTC',
                                                                                                                           'Volume USD':'VOLUME_USD',}).sort_values('TIMESTAMP', 
                                                                                                                                                                     ascending=True).reset_index(drop=True)
        current_df_3['DATE'] = current_df_3['TIMESTAMP'].apply(lambda x: datetime.utcfromtimestamp(x))

        print(current_df_3[current_df_3['TIMESTAMP'] == 1485216000])

        if f == '1_Day':
            file_name_2 = os.listdir(f"{base_file_path_2}/{cc}/{f}/")[0]
            current_df_2 = pd.read_csv(f"{base_file_path_2}/{cc}/{f}/{file_name_2}").drop(['time',
                                                                                           'id',
                                                                                           'symbol'], axis=1).rename(columns={'timeDate':'DATE',
                                                                                                                              'close':'CLOSE_PRICE',
                                                                                                                              'high':'HIGH_PRICE',
                                                                                                                              'low':'LOW_PRICE',
                                                                                                                              'open':'OPEN_PRICE',
                                                                                                                              'ts':'TIMESTAMP',
                                                                                                                              'volumefrom':'VOLUME_FROM',
                                                                                                                              'volumeto':'VOLUME_TO',
                                                                                                                              'zero_balance_addresses_all_time':'ALL_TIME_ZERO_BALANCE_ADDRESS_COUNT',
                                                                                                                              'unique_addresses_all_time':'ALL_TIME_UNIQUE_ADDRESS_COUNT',
                                                                                                                              'new_addresses':'NEW_ADDRESS_COUNT',
                                                                                                                              'active_addresses':'ACTIVE_ADDRESS_COUNT',
                                                                                                                              'new_addresses':'NEW_ADDRESS_COUNT',
                                                                                                                              'transaction_count':'TRANSACTION_COUNT',
                                                                                                                              'transaction_count_all_time':'ALL_TIME_TRANSACTION_COUNT',
                                                                                                                              'large_transaction_count':'LARGE_TRANSACTION_COUNT',
                                                                                                                              'average_transaction_value':'AVERAGE_TRANSACTION_VALUE',
                                                                                                                              'block_height':'BLOCK_HEIGHT',
                                                                                                                              'hashrate':'HASH_RATE',
                                                                                                                              'difficulty':'DIFFICULTY',
                                                                                                                              'block_time':'BLOCK_TIME',
                                                                                                                              'block_size':'BLOCK_SIZE',
                                                                                                                              'current_supply':'CURRENT_SUPPLY',})
            print(current_df_2[current_df_2['TIMESTAMP'] == 1485216000])
            print(current_df_2.columns)
            # print(current_df_2)
            #print(current_df_2[''])
        

        #file_name_1 = os.listdir(f"{base_file_path_1}/{cc}/{f}/")[0]
        
        
        # file_name_2 = os.listdir(f"{base_file_path_2}/{cc}/{f}/")[0]
        # current_df_2 = pd.read_csv(f"{base_file_path_2}/{cc}/{f}/{file_name_2}").drop(['symbol'], axis=1).rename(columns={'unix':'TIMESTAMP',
        #                                                                                                                   'date':'DATE',
        #                                                                                                                   'open':'OPEN_PRICE',
        #                                                                                                                   'high':'HIGH_PRICE',
        #                                                                                                                   'low':'LOW_PRICE',
        #                                                                                                                   'close':'CLOSE_PRICE',
        #                                                                                                                   'Volume BTC':'VOLUME_BTC',
        #                                                                                                                   'Volume USDT':'VOLUME_USDT',
        #                                                                                                                   'buyTakerQuantity':'VOLUME_BUY',
        #                                                                                                                   'buyTakerAmount':'QUOTE_VOLUME_BUY',
        #                                                                                                                   'tradeCount':'TOTAL_TRADE_COUNT',
        #                                                                                                                   'weightedAverage':'WEIGHTED_AVERAGE_PRICE'}).sort_values('TIMESTAMP', 
        #                                                                                                                                                                            ascending=True).reset_index(drop=True)
        # current_df_2['TIMESTAMP'] = current_df_2['TIMESTAMP'].apply(lambda x: int(x / 1000))
        # current_df_2['DATE'] = pd.to_datetime(current_df_2['DATE'])

        # merged_df = pd.merge(current_df_1, 
        #                      current_df_2, 
        #                      on='TIMESTAMP', 
        #                      how='outer').sort_values('TIMESTAMP', 
        #                                               ascending=True).reset_index(drop=True)

        # merged_df['DATE'] = merged_df['DATE_x'].combine_first(merged_df['DATE_y'])
        # merged_df = merged_df.drop(['DATE_x', 'DATE_y'], axis=1)

        # columns_in_common = [c for c in merged_df.columns if c.endswith('_x') or c.endswith('_y')] 
        # x_set = {col[:-2] for col in columns_in_common if col.endswith('_x')}
        # y_set = {col[:-2] for col in columns_in_common if col.endswith('_y')}
        # paired = sorted(x_set & y_set)
        # xy_pairs = [(f"{col}_x", f"{col}_y") for col in paired]

        # for c_x, c_y in xy_pairs:
        #     current_combined_column_name = c_x[:-2]
        #     current_combined_column_vals = []

        #     for x, y in zip(list(merged_df[c_x]), list(merged_df[c_y])):
        #         if (not pd.isna(x)) and (not pd.isna(y)):
        #             if current_combined_column_name == "LOW_PRICE":
        #                 current_combined_column_vals.append(min(x,y))
        #             elif (current_combined_column_name == "OPEN_PRICE") or (current_combined_column_name == "CLOSE_PRICE"):
        #                 current_combined_column_vals.append(sum([x,y]) / 2)
        #             else:
        #                 current_combined_column_vals.append(max(x,y))
        #         elif pd.isna(x) and pd.isna(y):
        #             current_combined_column_vals.append(x)
        #         elif pd.isna(x):
        #             current_combined_column_vals.append(y)
        #         else:
        #             current_combined_column_vals.append(x)
            
        #     merged_df[current_combined_column_name] = current_combined_column_vals
        #     merged_df = merged_df.drop([c_x, c_y], axis=1)

        # save_map_path = f"All_Crypto_Data/Crypto_Market_Data/All_Poloniex/{cc}/{f}/"

        # if not os.path.exists(save_map_path):
        #     os.makedirs(save_map_path)
        
        # merged_df.to_csv(f"{save_map_path}Poloniex_{cc}_USDT_UTC_Market_Data_{merged_df.iloc[0]['DATE'].strftime('%d_%m_%Y')}__{merged_df.iloc[-1]['DATE'].strftime('%d_%m_%Y')}.csv")

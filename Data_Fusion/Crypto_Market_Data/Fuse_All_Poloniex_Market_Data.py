import pandas as pd
import os
from datetime import datetime

base_file_path_1 = "All_Crypto_Data/Crypto_Market_Data/Poloniex_via_CoinDesk/"
base_file_path_2 = "All_Crypto_Data/Crypto_Market_Data/Poloniex_via_CryptoDataDownload/"

frequencies = ['1_Day', '1_Hour']

for cc in [m for m in os.listdir(base_file_path_1) if not m.endswith('.txt')]:
    for f in frequencies:
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
                                                      'VOLUME_UNKNOWN'], axis=1).rename(columns={'OPEN':'OPEN_PRICE_USDT',
                                                                                                 'CLOSE':'CLOSE_PRICE_USDT',
                                                                                                 'HIGH':'HIGH_PRICE_USDT',
                                                                                                 'LOW':'LOW_PRICE_USDT',
                                                                                                 'FIRST_TRADE_PRICE':'FIRST_TRADE_PRICE_USDT',
                                                                                                 'HIGH_TRADE_PRICE':'HIGH_TRADE_PRICE_USDT',
                                                                                                 'LOW_TRADE_PRICE':'LOW_TRADE_PRICE_USDT',
                                                                                                 'LAST_TRADE_PRICE':'LAST_TRADE_PRICE_USDT',
                                                                                                 'VOLUME':'VOLUME_USDT',
                                                                                                 'TOTAL_TRADES':'TOTAL_TRADE_COUNT'}).sort_values('TIMESTAMP', 
                                                                                                                                                  ascending=True).reset_index(drop=True)
        current_df_1['DATE'] = current_df_1['TIMESTAMP'].apply(lambda x: datetime.utcfromtimestamp(x))
        
        file_name_2 = os.listdir(f"{base_file_path_2}/{cc}/{f}/")[0]
        current_df_2 = pd.read_csv(f"{base_file_path_2}/{cc}/{f}/{file_name_2}").drop(['symbol'], axis=1).rename(columns={'unix':'TIMESTAMP',
                                                                                                                          'date':'DATE',
                                                                                                                          'open':'OPEN_PRICE_USDT',
                                                                                                                          'high':'HIGH_PRICE_USDT',
                                                                                                                          'low':'LOW_PRICE_USDT',
                                                                                                                          'close':'CLOSE_PRICE_USDT',
                                                                                                                          'Volume BTC':'VOLUME_BTC',
                                                                                                                          'Volume USDT':'VOLUME_USDT',
                                                                                                                          'buyTakerQuantity':'VOLUME_BUY',
                                                                                                                          'buyTakerAmount':'QUOTE_VOLUME_BUY',
                                                                                                                          'tradeCount':'TOTAL_TRADE_COUNT',
                                                                                                                          'weightedAverage':'WEIGHTED_AVERAGE_PRICE_USDT'}).sort_values('TIMESTAMP', 
                                                                                                                                                                                        ascending=True).reset_index(drop=True)
        current_df_2['TIMESTAMP'] = current_df_2['TIMESTAMP'].apply(lambda x: int(x / 1000))
        current_df_2['DATE'] = pd.to_datetime(current_df_2['DATE'])

        merged_df = pd.merge(current_df_1, 
                             current_df_2, 
                             on='TIMESTAMP', 
                             how='outer').sort_values('TIMESTAMP', 
                                                      ascending=True).reset_index(drop=True)

        merged_df['DATE'] = merged_df['DATE_x'].combine_first(merged_df['DATE_y'])
        merged_df = merged_df.drop(['DATE_x', 'DATE_y'], axis=1)

        columns_in_common = [c for c in merged_df.columns if c.endswith('_x') or c.endswith('_y')] 
        x_set = {col[:-2] for col in columns_in_common if col.endswith('_x')}
        y_set = {col[:-2] for col in columns_in_common if col.endswith('_y')}
        paired = sorted(x_set & y_set)
        xy_pairs = [(f"{col}_x", f"{col}_y") for col in paired]

        for c_x, c_y in xy_pairs:
            current_combined_column_name = c_x[:-2]
            current_combined_column_vals = []

            for x, y in zip(list(merged_df[c_x]), list(merged_df[c_y])):
                if (not pd.isna(x)) and (not pd.isna(y)):
                    if current_combined_column_name == "LOW_PRICE_USDT":
                        current_combined_column_vals.append(min(x,y))
                    elif (current_combined_column_name == "OPEN_PRICE_USDT") or (current_combined_column_name == "CLOSE_PRICE_USDT"):
                        current_combined_column_vals.append(sum([x,y]) / 2)
                    else:
                        current_combined_column_vals.append(max(x,y))
                elif pd.isna(x) and pd.isna(y):
                    current_combined_column_vals.append(x)
                elif pd.isna(x):
                    current_combined_column_vals.append(y)
                else:
                    current_combined_column_vals.append(x)
            
            merged_df[current_combined_column_name] = current_combined_column_vals
            merged_df = merged_df.drop([c_x, c_y], axis=1)

        save_map_path = f"All_Crypto_Data/Crypto_Market_Data/All_Poloniex/{cc}/{f}/"

        if not os.path.exists(save_map_path):
            os.makedirs(save_map_path)
        
        merged_df.to_csv(f"{save_map_path}Poloniex_{cc}_USDT_UTC_Market_Data_{merged_df.iloc[0]['DATE'].strftime('%d_%m_%Y')}__{merged_df.iloc[-1]['DATE'].strftime('%d_%m_%Y')}.csv")

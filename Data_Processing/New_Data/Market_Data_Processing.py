import pandas as pd
import os
from datetime import datetime, timedelta

unprocessed_all_market_data_base_path = "All_Crypto_Data/New_Data/Unprocessed/Market_Data/All_Market_Data/"
unprocessed_only_binance_market_data_base_path = "All_Crypto_Data/New_Data/Unprocessed/Market_Data/Only_Binance_Market_Data/"

selected_ccs = ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']
selected_start_date = datetime(2017, 6, 1)
selected_end_date = datetime(2024, 12, 31, 23)


for cc in os.listdir(unprocessed_only_binance_market_data_base_path):
    for f in os.listdir(f"{unprocessed_only_binance_market_data_base_path}/{cc}/"):
        current_only_binance_market_data_file = os.listdir(f"{unprocessed_only_binance_market_data_base_path}/{cc}/{f}/")[0]
        current_only_binance_market_data_df = pd.read_csv(f"{unprocessed_only_binance_market_data_base_path}/{cc}/{f}/{current_only_binance_market_data_file}")
        current_only_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_only_binance_market_data_df['TIMESTAMP'])

        if f == '1_Day':
            current_only_binance_market_data_df = current_only_binance_market_data_df[['TIMESTAMP', f'{cc}_OPEN_PRICE_USD', f'{cc}_HIGH_PRICE_USD', f'{cc}_LOW_PRICE_USD', f'{cc}_CLOSE_PRICE_USD']] # Only keep these columns due to other columns not being available for first dates
            current_only_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_only_binance_market_data_df['TIMESTAMP'])

            current_start_date = max(current_only_binance_market_data_df['TIMESTAMP'].min(), selected_start_date)
            current_end_date = min(current_only_binance_market_data_df['TIMESTAMP'].max(), selected_end_date)
            current_date_range = pd.date_range(current_start_date,
                                               current_end_date,
                                               freq='D')
            current_date_range_df = pd.DataFrame({'TIMESTAMP':current_date_range})

            current_only_binance_market_data_df = pd.merge(current_date_range_df, 
                                                           current_only_binance_market_data_df,
                                                           on='TIMESTAMP',
                                                           how='left')
        else:
            current_only_binance_market_data_df = current_only_binance_market_data_df.rename(columns={f'{cc}_VOLUME_USD':f'{cc}_VOLUME_{cc}',
                                                                                                      f'{cc}_QUOTE_VOLUME':f'{cc}_VOLUME_USD',
                                                                                                      f'{cc}_VOLUME_BUY':f'{cc}_BUY_VOLUME_{cc}',
                                                                                                      f'{cc}_QUOTE_VOLUME_BUY':f'{cc}_BUY_VOLUME_USD',
                                                                                                      f'{cc}_VOLUME_SELL':f'{cc}_SELL_VOLUME_{cc}',
                                                                                                      f'{cc}_QUOTE_VOLUME_SELL':f'{cc}_SELL_VOLUME_USD',})

        if current_only_binance_market_data_df.isna().any().any():
            if f != '1_Day':
                current_only_binance_market_data_df[f'{cc}_FIRST_TRADE_PRICE_USD'] = current_only_binance_market_data_df[f'{cc}_FIRST_TRADE_PRICE_USD'].fillna(current_only_binance_market_data_df[f'{cc}_OPEN_PRICE_USD'])
                current_only_binance_market_data_df[f'{cc}_HIGH_TRADE_PRICE_USD'] = current_only_binance_market_data_df[f'{cc}_HIGH_TRADE_PRICE_USD'].fillna(current_only_binance_market_data_df[f'{cc}_HIGH_PRICE_USD'])
                current_only_binance_market_data_df[f'{cc}_LOW_TRADE_PRICE_USD'] = current_only_binance_market_data_df[f'{cc}_LOW_TRADE_PRICE_USD'].fillna(current_only_binance_market_data_df[f'{cc}_LOW_PRICE_USD'])
                current_only_binance_market_data_df[f'{cc}_LAST_TRADE_PRICE_USD'] = current_only_binance_market_data_df[f'{cc}_LAST_TRADE_PRICE_USD'].fillna(current_only_binance_market_data_df[f'{cc}_CLOSE_PRICE_USD'])
        
        current_processed_only_binance_market_data_path = f'All_Crypto_Data/New_Data/Processed/Market_Data/Only_Binance_Market_Data/{cc}/{f}/'

        if not os.path.exists(current_processed_only_binance_market_data_path):
            os.makedirs(current_processed_only_binance_market_data_path)
        
        if f == '1_Day':
            current_only_binance_market_data_df.to_csv(f"{current_processed_only_binance_market_data_path}Binance_{cc}_USD_Daily_Market_Data_{current_only_binance_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_only_binance_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv",
                                                       index=False)
        elif f == '1_Hour':
            current_only_binance_market_data_df.to_csv(f"{current_processed_only_binance_market_data_path}Binance_{cc}_USD_Hourly_Market_Data_{current_only_binance_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_only_binance_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv",
                                                       index=False)
        else:
            current_only_binance_market_data_df.to_csv(f"{current_processed_only_binance_market_data_path}Binance_{cc}_USD_{f}ly_Market_Data_{current_only_binance_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_only_binance_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv",
                                                       index=False)


for cc in os.listdir(unprocessed_all_market_data_base_path):
    for f in os.listdir(f"{unprocessed_all_market_data_base_path}/{cc}/"):
        current_all_market_data_file = os.listdir(f"{unprocessed_all_market_data_base_path}/{cc}/{f}/")[0]
        current_alle_market_data_df = pd.read_csv(f"{unprocessed_all_market_data_base_path}/{cc}/{f}/{current_all_market_data_file}")
        current_alle_market_data_df['TIMESTAMP'] = pd.to_datetime(current_alle_market_data_df['TIMESTAMP'])

        if f == '1_Day':
            current_alle_market_data_df = current_alle_market_data_df.drop(columns=[f'{cc}_FIRST_TRADE_PRICE_USD',
                                                                                    f'{cc}_HIGH_TRADE_PRICE_USD',
                                                                                    f'{cc}_LOW_TRADE_PRICE_USD',
                                                                                    f'{cc}_LAST_TRADE_PRICE_USD',
                                                                                    f'{cc}_TOTAL_TRADE_COUNT',
                                                                                    f'{cc}_TOTAL_TRADES_BUY',
                                                                                    f'{cc}_TOTAL_TRADES_SELL',
                                                                                    f'{cc}_VOLUME_USD',
                                                                                    f'{cc}_QUOTE_VOLUME',
                                                                                    f'{cc}_VOLUME_BUY',
                                                                                    f'{cc}_QUOTE_VOLUME_BUY',
                                                                                    f'{cc}_VOLUME_SELL',
                                                                                    f'{cc}_QUOTE_VOLUME_SELL',], 
                                                                                    axis=1) # Drop these columns due to columns not being available for first few dates
            
            current_start_date = max(current_alle_market_data_df['TIMESTAMP'].min(), selected_start_date)
            current_end_date = min(current_alle_market_data_df['TIMESTAMP'].max(), selected_end_date)
            current_date_range = pd.date_range(current_start_date,
                                               current_end_date,
                                               freq='D')
            current_date_range_df = pd.DataFrame({'TIMESTAMP':current_date_range})

            current_alle_market_data_df = pd.merge(current_date_range_df, 
                                                   current_alle_market_data_df,
                                                   on='TIMESTAMP',
                                                   how='left')
        else:
            current_alle_market_data_df = current_alle_market_data_df.rename(columns={f'{cc}_VOLUME_USD':f'{cc}_VOLUME_{cc}',
                                                                                      f'{cc}_QUOTE_VOLUME':f'{cc}_VOLUME_USD',
                                                                                      f'{cc}_VOLUME_BUY':f'{cc}_BUY_VOLUME_{cc}',
                                                                                      f'{cc}_QUOTE_VOLUME_BUY':f'{cc}_BUY_VOLUME_USD',
                                                                                      f'{cc}_VOLUME_SELL':f'{cc}_SELL_VOLUME_{cc}',
                                                                                      f'{cc}_QUOTE_VOLUME_SELL':f'{cc}_SELL_VOLUME_USD',})

        if current_alle_market_data_df.isna().any().any():
            if f != '1_Day':
                current_alle_market_data_df[f'{cc}_FIRST_TRADE_PRICE_USD'] = current_alle_market_data_df[f'{cc}_FIRST_TRADE_PRICE_USD'].fillna(current_alle_market_data_df[f'{cc}_OPEN_PRICE_USD'])
                current_alle_market_data_df[f'{cc}_HIGH_TRADE_PRICE_USD'] = current_alle_market_data_df[f'{cc}_HIGH_TRADE_PRICE_USD'].fillna(current_alle_market_data_df[f'{cc}_HIGH_PRICE_USD'])
                current_alle_market_data_df[f'{cc}_LOW_TRADE_PRICE_USD'] = current_alle_market_data_df[f'{cc}_LOW_TRADE_PRICE_USD'].fillna(current_alle_market_data_df[f'{cc}_LOW_PRICE_USD'])
                current_alle_market_data_df[f'{cc}_LAST_TRADE_PRICE_USD'] = current_alle_market_data_df[f'{cc}_LAST_TRADE_PRICE_USD'].fillna(current_alle_market_data_df[f'{cc}_CLOSE_PRICE_USD'])

                if cc == 'BTC':
                    current_alle_market_data_df = current_alle_market_data_df.drop(['BTC_MARKET_CAP_USD_PER_BLOCKCHAIN', 
                                                                                    'BTC_SUPPLY_PER_BLOCKCHAIN'], 
                                                                                    axis=1) # Drop due to NaN values and features not being part of data of other ccs
            else:
                if cc not in ['BTC', 'ETH']:
                    current_alle_market_data_df[f'{cc}_MARKET_CAP_USD_PER_BIC'] = current_alle_market_data_df[f'{cc}_MARKET_CAP_USD_PER_BIC'].fillna(current_alle_market_data_df[f'{cc}_MARKET_CAP_USD_PER_CMC'])
                
                if cc == 'BTC':
                    daily_blockchain_df = pd.read_csv('All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain/BTC/1_Day/Blockchain_BTC_USD_Daily_Market_Data_03_01_2009__01_06_2025.csv')
                    daily_blockchain_df['TIMESTAMP'] = pd.to_datetime(daily_blockchain_df['TIMESTAMP'])

                    current_alle_market_data_df['BTC_MARKET_CAP_USD_PER_BLOCKCHAIN_VIA_QUANDL'] = current_alle_market_data_df.apply(lambda row: daily_blockchain_df[daily_blockchain_df['TIMESTAMP'] == row.TIMESTAMP].iloc[0]['BTC_MARKET_CAP_USD'] if pd.isna(row.BTC_MARKET_CAP_USD_PER_BLOCKCHAIN_VIA_QUANDL) else row.BTC_MARKET_CAP_USD_PER_BLOCKCHAIN_VIA_QUANDL,
                                                                                                                                    axis=1)
                    current_alle_market_data_df['BTC_SUPPLY_PER_BLOCKCHAIN_VIA_QUANDL'] = current_alle_market_data_df.apply(lambda row: daily_blockchain_df[daily_blockchain_df['TIMESTAMP'] == row.TIMESTAMP].iloc[0]['BTC_TOTAL_CIRCULATING_COINS'] if pd.isna(row.BTC_SUPPLY_PER_BLOCKCHAIN_VIA_QUANDL) else row.BTC_SUPPLY_PER_BLOCKCHAIN_VIA_QUANDL,
                                                                                                                            axis=1)
                
        current_processed_all_market_data_path = f'All_Crypto_Data/New_Data/Processed/Market_Data/All_Market_Data/{cc}/{f}/'

        if not os.path.exists(current_processed_all_market_data_path):
            os.makedirs(current_processed_all_market_data_path)
        
        if f == '1_Day':
            current_alle_market_data_df.to_csv(f"{current_processed_all_market_data_path}All_Market_Data_{cc}_USD_Daily_{current_alle_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_alle_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv",
                                               index=False)
        elif f == '1_Hour':
            current_alle_market_data_df.to_csv(f"{current_processed_all_market_data_path}All_Market_Data_{cc}_USD_Hourly_{current_alle_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_alle_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv",
                                               index=False)
        else:
            current_alle_market_data_df.to_csv(f"{current_processed_all_market_data_path}All_Market_Data_{cc}_USD_{f}ly_{current_alle_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_alle_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv",
                                               index=False)


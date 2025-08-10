import pandas as pd
import os
from datetime import datetime

daily_binance_market_data_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Binance_Extended/"
hourly_binance_market_data_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Binance_via_CoinDesk/"


for cc in os.listdir(daily_binance_market_data_path):
    current_daily_binance_market_data_fp = f"{daily_binance_market_data_path}{cc}/1_Day/"
    current_12_hourly_binance_market_data_fp = f"{hourly_binance_market_data_path}{cc}/12_Hour/"
    current_8_hourly_binance_market_data_fp = f"{hourly_binance_market_data_path}{cc}/8_Hour/"
    current_6_hourly_binance_market_data_fp = f"{hourly_binance_market_data_path}{cc}/6_Hour/"
    current_4_hourly_binance_market_data_fp = f"{hourly_binance_market_data_path}{cc}/4_Hour/"
    current_2_hourly_binance_market_data_fp = f"{hourly_binance_market_data_path}{cc}/2_Hour/"
    current_hourly_binance_market_data_fp = f"{hourly_binance_market_data_path}{cc}/1_Hour/"

    current_daily_binance_market_data_fn = os.listdir(current_daily_binance_market_data_fp)[0]
    current_12_hourly_binance_market_data_fn = os.listdir(current_12_hourly_binance_market_data_fp)[0]
    current_8_hourly_binance_market_data_fn = os.listdir(current_8_hourly_binance_market_data_fp)[0]
    current_6_hourly_binance_market_data_fn = os.listdir(current_6_hourly_binance_market_data_fp)[0]
    current_4_hourly_binance_market_data_fn = os.listdir(current_4_hourly_binance_market_data_fp)[0]
    current_2_hourly_binance_market_data_fn = os.listdir(current_2_hourly_binance_market_data_fp)[0]
    current_hourly_binance_market_data_fn = os.listdir(current_hourly_binance_market_data_fp)[0]

    current_daily_binance_market_data_df = pd.read_csv(f"{current_daily_binance_market_data_fp}{current_daily_binance_market_data_fn}")
    current_daily_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_daily_binance_market_data_df['TIMESTAMP'])

    current_12_hourly_binance_market_data_df = pd.read_csv(f"{current_12_hourly_binance_market_data_fp}{current_12_hourly_binance_market_data_fn}")
    current_12_hourly_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_12_hourly_binance_market_data_df['TIMESTAMP'])

    current_8_hourly_binance_market_data_df = pd.read_csv(f"{current_8_hourly_binance_market_data_fp}{current_8_hourly_binance_market_data_fn}")
    current_8_hourly_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_8_hourly_binance_market_data_df['TIMESTAMP'])

    current_6_hourly_binance_market_data_df = pd.read_csv(f"{current_6_hourly_binance_market_data_fp}{current_6_hourly_binance_market_data_fn}")
    current_6_hourly_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_6_hourly_binance_market_data_df['TIMESTAMP'])

    current_4_hourly_binance_market_data_df = pd.read_csv(f"{current_4_hourly_binance_market_data_fp}{current_4_hourly_binance_market_data_fn}")
    current_4_hourly_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_4_hourly_binance_market_data_df['TIMESTAMP'])

    current_2_hourly_binance_market_data_df = pd.read_csv(f"{current_2_hourly_binance_market_data_fp}{current_2_hourly_binance_market_data_fn}")
    current_2_hourly_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_2_hourly_binance_market_data_df['TIMESTAMP'])

    current_hourly_binance_market_data_df = pd.read_csv(f"{current_hourly_binance_market_data_fp}{current_hourly_binance_market_data_fn}")
    current_hourly_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_hourly_binance_market_data_df['TIMESTAMP'])

    selected_cs = [c for c in current_hourly_binance_market_data_df.columns if 'USDT' in c]
    new_names_cs = [c.replace('USDT', 'USD') for c in selected_cs]

    current_12_hourly_binance_market_data_df = current_12_hourly_binance_market_data_df.rename(columns=dict(list(zip(selected_cs, new_names_cs))))
    current_8_hourly_binance_market_data_df = current_8_hourly_binance_market_data_df.rename(columns=dict(list(zip(selected_cs, new_names_cs))))
    current_6_hourly_binance_market_data_df = current_6_hourly_binance_market_data_df.rename(columns=dict(list(zip(selected_cs, new_names_cs))))
    current_4_hourly_binance_market_data_df = current_4_hourly_binance_market_data_df.rename(columns=dict(list(zip(selected_cs, new_names_cs))))
    current_2_hourly_binance_market_data_df = current_2_hourly_binance_market_data_df.rename(columns=dict(list(zip(selected_cs, new_names_cs))))
    current_hourly_binance_market_data_df = current_hourly_binance_market_data_df.rename(columns=dict(list(zip(selected_cs, new_names_cs))))
    
    current_daily_only_binance_market_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Market_Data/Only_Binance_Market_Data/{cc}/1_Day/"
    current_12_hourly_only_binance_market_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Market_Data/Only_Binance_Market_Data/{cc}/12_Hour/"
    current_8_hourly_only_binance_market_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Market_Data/Only_Binance_Market_Data/{cc}/8_Hour/"
    current_6_hourly_only_binance_market_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Market_Data/Only_Binance_Market_Data/{cc}/6_Hour/"
    current_4_hourly_only_binance_market_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Market_Data/Only_Binance_Market_Data/{cc}/4_Hour/"
    current_2_hourly_only_binance_market_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Market_Data/Only_Binance_Market_Data/{cc}/2_Hour/"
    current_hourly_only_binance_market_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Market_Data/Only_Binance_Market_Data/{cc}/1_Hour/"
    
    if not os.path.exists(current_daily_only_binance_market_data_fp):
        os.makedirs(current_daily_only_binance_market_data_fp)

    if not os.path.exists(current_12_hourly_only_binance_market_data_fp):
        os.makedirs(current_12_hourly_only_binance_market_data_fp)

    if not os.path.exists(current_8_hourly_only_binance_market_data_fp):
        os.makedirs(current_8_hourly_only_binance_market_data_fp)
    
    if not os.path.exists(current_6_hourly_only_binance_market_data_fp):
        os.makedirs(current_6_hourly_only_binance_market_data_fp)
    
    if not os.path.exists(current_4_hourly_only_binance_market_data_fp):
        os.makedirs(current_4_hourly_only_binance_market_data_fp)
    
    if not os.path.exists(current_2_hourly_only_binance_market_data_fp):
        os.makedirs(current_2_hourly_only_binance_market_data_fp)
    
    if not os.path.exists(current_hourly_only_binance_market_data_fp):
        os.makedirs(current_hourly_only_binance_market_data_fp)

    current_daily_binance_market_data_df.to_csv(f"{current_daily_only_binance_market_data_fp}Binance_{cc}_USD_Daily_Market_Data_{current_daily_binance_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_daily_binance_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                                index=False)
    current_12_hourly_binance_market_data_df.to_csv(f"{current_12_hourly_only_binance_market_data_fp}Binance_{cc}_USD_Hourly_Market_Data_{current_12_hourly_binance_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_12_hourly_binance_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv",
                                                    index=False)
    current_8_hourly_binance_market_data_df.to_csv(f"{current_8_hourly_only_binance_market_data_fp}Binance_{cc}_USD_Hourly_Market_Data_{current_8_hourly_binance_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_8_hourly_binance_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv",
                                                   index=False)
    current_6_hourly_binance_market_data_df.to_csv(f"{current_6_hourly_only_binance_market_data_fp}Binance_{cc}_USD_Hourly_Market_Data_{current_6_hourly_binance_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_6_hourly_binance_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv",
                                                   index=False)
    current_4_hourly_binance_market_data_df.to_csv(f"{current_4_hourly_only_binance_market_data_fp}Binance_{cc}_USD_Hourly_Market_Data_{current_4_hourly_binance_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_4_hourly_binance_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv",
                                                 index=False)
    current_2_hourly_binance_market_data_df.to_csv(f"{current_2_hourly_only_binance_market_data_fp}Binance_{cc}_USD_Hourly_Market_Data_{current_2_hourly_binance_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_2_hourly_binance_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv",
                                                   index=False)
    current_hourly_binance_market_data_df.to_csv(f"{current_hourly_only_binance_market_data_fp}Binance_{cc}_USD_Hourly_Market_Data_{current_hourly_binance_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_hourly_binance_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv",
                                                 index=False)


only_binance_market_data_path = "All_Crypto_Data/New_Data/Unprocessed/Market_Data/Only_Binance_Market_Data/"
cmc_data_base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/CoinMarketCap/"

for cc in os.listdir(only_binance_market_data_path):
    current_daily_only_binance_market_data_fp = f"{only_binance_market_data_path}{cc}/1_Day/"
    current_daily_only_binance_market_data_fn = os.listdir(current_daily_only_binance_market_data_fp)[0]
    current_daily_only_binance_market_data_df = pd.read_csv(f"{current_daily_only_binance_market_data_fp}{current_daily_only_binance_market_data_fn}")
    current_daily_only_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_daily_only_binance_market_data_df['TIMESTAMP'])
    
    current_12_hourly_only_binance_market_data_fp = f"{only_binance_market_data_path}{cc}/12_Hour/"
    current_12_hourly_only_binance_market_data_fn = os.listdir(current_12_hourly_only_binance_market_data_fp)[0]
    current_12_hourly_only_binance_market_data_df = pd.read_csv(f"{current_12_hourly_only_binance_market_data_fp}{current_12_hourly_only_binance_market_data_fn}")
    current_12_hourly_only_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_12_hourly_only_binance_market_data_df['TIMESTAMP'])

    current_8_hourly_only_binance_market_data_fp = f"{only_binance_market_data_path}{cc}/8_Hour/"
    current_8_hourly_only_binance_market_data_fn = os.listdir(current_8_hourly_only_binance_market_data_fp)[0]
    current_8_hourly_only_binance_market_data_df = pd.read_csv(f"{current_8_hourly_only_binance_market_data_fp}{current_8_hourly_only_binance_market_data_fn}")
    current_8_hourly_only_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_8_hourly_only_binance_market_data_df['TIMESTAMP'])

    current_6_hourly_only_binance_market_data_fp = f"{only_binance_market_data_path}{cc}/6_Hour/"
    current_6_hourly_only_binance_market_data_fn = os.listdir(current_6_hourly_only_binance_market_data_fp)[0]
    current_6_hourly_only_binance_market_data_df = pd.read_csv(f"{current_6_hourly_only_binance_market_data_fp}{current_6_hourly_only_binance_market_data_fn}")
    current_6_hourly_only_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_6_hourly_only_binance_market_data_df['TIMESTAMP'])

    current_4_hourly_only_binance_market_data_fp = f"{only_binance_market_data_path}{cc}/4_Hour/"
    current_4_hourly_only_binance_market_data_fn = os.listdir(current_4_hourly_only_binance_market_data_fp)[0]
    current_4_hourly_only_binance_market_data_df = pd.read_csv(f"{current_4_hourly_only_binance_market_data_fp}{current_4_hourly_only_binance_market_data_fn}")
    current_4_hourly_only_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_4_hourly_only_binance_market_data_df['TIMESTAMP'])

    current_2_hourly_only_binance_market_data_fp = f"{only_binance_market_data_path}{cc}/2_Hour/"
    current_2_hourly_only_binance_market_data_fn = os.listdir(current_2_hourly_only_binance_market_data_fp)[0]
    current_2_hourly_only_binance_market_data_df = pd.read_csv(f"{current_2_hourly_only_binance_market_data_fp}{current_2_hourly_only_binance_market_data_fn}")
    current_2_hourly_only_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_2_hourly_only_binance_market_data_df['TIMESTAMP'])

    current_hourly_only_binance_market_data_fp = f"{only_binance_market_data_path}{cc}/1_Hour/"
    current_hourly_only_binance_market_data_fn = os.listdir(current_hourly_only_binance_market_data_fp)[0]
    current_hourly_only_binance_market_data_df = pd.read_csv(f"{current_hourly_only_binance_market_data_fp}{current_hourly_only_binance_market_data_fn}")
    current_hourly_only_binance_market_data_df['TIMESTAMP'] = pd.to_datetime(current_hourly_only_binance_market_data_df['TIMESTAMP'])

    if cc == 'BTC':
        daily_btc_blockchain_via_quandl_market_data_df = pd.read_csv("All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain_via_Quandl/BTC/1_Day/Blockchain_via_Quandl_BTC_USD_Daily_Market_Data_02_01_2009__12_06_2025.csv")
        daily_btc_blockchain_via_quandl_market_data_df['TIMESTAMP'] = pd.to_datetime(daily_btc_blockchain_via_quandl_market_data_df['TIMESTAMP'])
        current_daily_all_market_data_df = pd.merge(current_daily_only_binance_market_data_df, 
                                                    daily_btc_blockchain_via_quandl_market_data_df[['TIMESTAMP', 'BTC_MARKET_CAP_USD', 'BTC_TOTAL_COINS']], 
                                                    on='TIMESTAMP', 
                                                    how='left').rename(columns={'BTC_TOTAL_COINS':'BTC_SUPPLY_PER_BLOCKCHAIN_VIA_QUANDL',
                                                                                'BTC_MARKET_CAP_USD':'BTC_MARKET_CAP_USD_PER_BLOCKCHAIN_VIA_QUANDL'})

        hourly_12_btc_blockchain_market_data_df = pd.read_csv("All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain/BTC/12_Hour/Blockchain_BTC_USD_12_Hourly_Market_Data_03_01_2009__01_06_2025.csv")
        hourly_12_btc_blockchain_market_data_df['TIMESTAMP'] = pd.to_datetime(hourly_12_btc_blockchain_market_data_df['TIMESTAMP'])
        current_12_hourly_all_market_data_df = pd.merge(current_12_hourly_only_binance_market_data_df, 
                                                        hourly_12_btc_blockchain_market_data_df, 
                                                        on='TIMESTAMP', 
                                                        how='left').rename(columns={'BTC_TOTAL_CIRCULATING_COINS':'BTC_SUPPLY_PER_BLOCKCHAIN',
                                                                                    'BTC_MARKET_CAP_USD':'BTC_MARKET_CAP_USD_PER_BLOCKCHAIN'})
        
        hourly_8_btc_blockchain_market_data_df = pd.read_csv("All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain/BTC/8_Hour/Blockchain_BTC_USD_8_Hourly_Market_Data_03_01_2009__01_06_2025.csv")
        hourly_8_btc_blockchain_market_data_df['TIMESTAMP'] = pd.to_datetime(hourly_8_btc_blockchain_market_data_df['TIMESTAMP'])
        current_8_hourly_all_market_data_df = pd.merge(current_8_hourly_only_binance_market_data_df, 
                                                       hourly_8_btc_blockchain_market_data_df, 
                                                       on='TIMESTAMP', 
                                                       how='left').rename(columns={'BTC_TOTAL_CIRCULATING_COINS':'BTC_SUPPLY_PER_BLOCKCHAIN',
                                                                                   'BTC_MARKET_CAP_USD':'BTC_MARKET_CAP_USD_PER_BLOCKCHAIN'})
        
        hourly_6_btc_blockchain_market_data_df = pd.read_csv("All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain/BTC/6_Hour/Blockchain_BTC_USD_6_Hourly_Market_Data_03_01_2009__01_06_2025.csv")
        hourly_6_btc_blockchain_market_data_df['TIMESTAMP'] = pd.to_datetime(hourly_6_btc_blockchain_market_data_df['TIMESTAMP'])
        current_6_hourly_all_market_data_df = pd.merge(current_6_hourly_only_binance_market_data_df, 
                                                       hourly_6_btc_blockchain_market_data_df, 
                                                       on='TIMESTAMP', 
                                                       how='left').rename(columns={'BTC_TOTAL_CIRCULATING_COINS':'BTC_SUPPLY_PER_BLOCKCHAIN',
                                                                                   'BTC_MARKET_CAP_USD':'BTC_MARKET_CAP_USD_PER_BLOCKCHAIN'})
        
        hourly_4_btc_blockchain_market_data_df = pd.read_csv("All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain/BTC/4_Hour/Blockchain_BTC_USD_4_Hourly_Market_Data_03_01_2009__01_06_2025.csv")
        hourly_4_btc_blockchain_market_data_df['TIMESTAMP'] = pd.to_datetime(hourly_4_btc_blockchain_market_data_df['TIMESTAMP'])
        current_4_hourly_all_market_data_df = pd.merge(current_4_hourly_only_binance_market_data_df, 
                                                       hourly_4_btc_blockchain_market_data_df, 
                                                       on='TIMESTAMP', 
                                                       how='left').rename(columns={'BTC_TOTAL_CIRCULATING_COINS':'BTC_SUPPLY_PER_BLOCKCHAIN',
                                                                                   'BTC_MARKET_CAP_USD':'BTC_MARKET_CAP_USD_PER_BLOCKCHAIN'})
        
        hourly_2_btc_blockchain_market_data_df = pd.read_csv("All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain/BTC/2_Hour/Blockchain_BTC_USD_2_Hourly_Market_Data_03_01_2009__01_06_2025.csv")
        hourly_2_btc_blockchain_market_data_df['TIMESTAMP'] = pd.to_datetime(hourly_2_btc_blockchain_market_data_df['TIMESTAMP'])
        current_2_hourly_all_market_data_df = pd.merge(current_2_hourly_only_binance_market_data_df, 
                                                       hourly_2_btc_blockchain_market_data_df, 
                                                       on='TIMESTAMP', 
                                                       how='left').rename(columns={'BTC_TOTAL_CIRCULATING_COINS':'BTC_SUPPLY_PER_BLOCKCHAIN',
                                                                                   'BTC_MARKET_CAP_USD':'BTC_MARKET_CAP_USD_PER_BLOCKCHAIN'})
        
        hourly_btc_blockchain_market_data_df = pd.read_csv("All_Crypto_Data/Crypto_Market_Data/Merged/Blockchain/BTC/1_Hour/Blockchain_BTC_USD_Hourly_Market_Data_03_01_2009__01_06_2025.csv")
        hourly_btc_blockchain_market_data_df['TIMESTAMP'] = pd.to_datetime(hourly_btc_blockchain_market_data_df['TIMESTAMP'])
        current_hourly_all_market_data_df = pd.merge(current_hourly_only_binance_market_data_df, 
                                                     hourly_btc_blockchain_market_data_df, 
                                                     on='TIMESTAMP', 
                                                     how='left').rename(columns={'BTC_TOTAL_CIRCULATING_COINS':'BTC_SUPPLY_PER_BLOCKCHAIN',
                                                                                 'BTC_MARKET_CAP_USD':'BTC_MARKET_CAP_USD_PER_BLOCKCHAIN'})
    elif cc == 'ETH':
        daily_eth_etherscan_market_data_df = pd.read_csv("All_Crypto_Data/Crypto_Market_Data/Merged/Etherscan/ETH/1_Day/Etherscan_ETH_USD_Daily_UTC_Market_Data_30_07_2015__14_05_2025.csv")
        daily_eth_etherscan_market_data_df['TIMESTAMP'] = pd.to_datetime(daily_eth_etherscan_market_data_df['TIMESTAMP'])
        current_daily_all_market_data_df = pd.merge(current_daily_only_binance_market_data_df, 
                                                    daily_eth_etherscan_market_data_df[['TIMESTAMP', 'ETH_MARKET_CAP_USD', 'ETH_SUPPLY']], 
                                                    on='TIMESTAMP', 
                                                    how='left').rename(columns={'ETH_MARKET_CAP_USD':'ETH_MARKET_CAP_USD_PER_ETHERSCAN',
                                                                                'ETH_SUPPLY':'ETH_SUPPLY_PER_ETHERSCAN'})

        current_12_hourly_all_market_data_df = current_12_hourly_only_binance_market_data_df
        current_8_hourly_all_market_data_df = current_8_hourly_only_binance_market_data_df
        current_6_hourly_all_market_data_df = current_6_hourly_only_binance_market_data_df
        current_4_hourly_all_market_data_df = current_4_hourly_only_binance_market_data_df
        current_2_hourly_all_market_data_df = current_2_hourly_only_binance_market_data_df
        current_hourly_all_market_data_df = current_hourly_only_binance_market_data_df
    else:
        bitinfocharts_market_data_fp = "All_Crypto_Data/Crypto_Market_Data/Merged/BitInfoCharts/"
        daily_bitinfocharts_market_data_fn = os.listdir(f"{bitinfocharts_market_data_fp}{cc}/1_Day/")[0]
        daily_bitinfocharts_market_data_df = pd.read_csv(f"{bitinfocharts_market_data_fp}{cc}/1_Day/{daily_bitinfocharts_market_data_fn}")
        daily_bitinfocharts_market_data_df['TIMESTAMP'] = pd.to_datetime(daily_bitinfocharts_market_data_df['TIMESTAMP'])
        current_daily_all_market_data_df = pd.merge(current_daily_only_binance_market_data_df, 
                                                    daily_bitinfocharts_market_data_df[['TIMESTAMP', f'{cc}_MARKET_CAP_USD']], 
                                                    on='TIMESTAMP', 
                                                    how='left').rename(columns={f'{cc}_MARKET_CAP_USD':f'{cc}_MARKET_CAP_USD_PER_BIC'})
        
        current_12_hourly_all_market_data_df = current_12_hourly_only_binance_market_data_df
        current_8_hourly_all_market_data_df = current_8_hourly_only_binance_market_data_df
        current_6_hourly_all_market_data_df = current_6_hourly_only_binance_market_data_df
        current_4_hourly_all_market_data_df = current_4_hourly_only_binance_market_data_df
        current_2_hourly_all_market_data_df = current_2_hourly_only_binance_market_data_df
        current_hourly_all_market_data_df = current_hourly_only_binance_market_data_df
    
    current_cmc_data_path = f"{cmc_data_base_path}/{cc}/1_Day/"
    current_cmc_data_file_name = os.listdir(current_cmc_data_path)[0]
    current_daily_cmc_df = pd.read_csv(f"{current_cmc_data_path}{current_cmc_data_file_name}")
    current_daily_cmc_df['TIMESTAMP'] = pd.to_datetime(current_daily_cmc_df['TIMESTAMP']).dt.tz_localize(None)

    current_daily_all_market_data_df = pd.merge(current_daily_all_market_data_df, 
                                                current_daily_cmc_df[['TIMESTAMP', f'{cc}_VOLUME_USD', f'{cc}_MARKET_CAP_USD']].rename(columns={f'{cc}_VOLUME_USD':f'{cc}_VOLUME_USD_PER_CMC',
                                                                                                                                                f'{cc}_MARKET_CAP_USD':f'{cc}_MARKET_CAP_USD_PER_CMC',}), 
                                                on='TIMESTAMP', 
                                                how='left')

    current_daily_all_market_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Market_Data/All_Market_Data/{cc}/1_Day/"
    current_12_hourly_all_market_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Market_Data/All_Market_Data/{cc}/12_Hour/"
    current_8_hourly_all_market_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Market_Data/All_Market_Data/{cc}/8_Hour/"
    current_6_hourly_all_market_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Market_Data/All_Market_Data/{cc}/6_Hour/"
    current_4_hourly_all_market_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Market_Data/All_Market_Data/{cc}/4_Hour/"
    current_2_hourly_all_market_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Market_Data/All_Market_Data/{cc}/2_Hour/"
    current_hourly_all_market_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Market_Data/All_Market_Data/{cc}/1_Hour/"
    
    if not os.path.exists(current_daily_all_market_data_fp):
        os.makedirs(current_daily_all_market_data_fp)

    if not os.path.exists(current_12_hourly_all_market_data_fp):
        os.makedirs(current_12_hourly_all_market_data_fp)
    
    if not os.path.exists(current_8_hourly_all_market_data_fp):
        os.makedirs(current_8_hourly_all_market_data_fp)
    
    if not os.path.exists(current_6_hourly_all_market_data_fp):
        os.makedirs(current_6_hourly_all_market_data_fp)
    
    if not os.path.exists(current_4_hourly_all_market_data_fp):
        os.makedirs(current_4_hourly_all_market_data_fp)
    
    if not os.path.exists(current_2_hourly_all_market_data_fp):
        os.makedirs(current_2_hourly_all_market_data_fp)
    
    if not os.path.exists(current_hourly_all_market_data_fp):
        os.makedirs(current_hourly_all_market_data_fp)
    
    current_daily_all_market_data_df.to_csv(f"{current_daily_all_market_data_fp}All_Market_Data_BTC_USD_Daily_{current_daily_all_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_daily_all_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                            index=False)
    current_12_hourly_all_market_data_df.to_csv(f"{current_12_hourly_all_market_data_fp}All_Market_Data_BTC_USD_12_Hourly_{current_12_hourly_all_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_12_hourly_all_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                                index=False)
    current_8_hourly_all_market_data_df.to_csv(f"{current_8_hourly_all_market_data_fp}All_Market_Data_BTC_USD_8_Hourly_{current_8_hourly_all_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_8_hourly_all_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                               index=False)
    current_6_hourly_all_market_data_df.to_csv(f"{current_6_hourly_all_market_data_fp}All_Market_Data_BTC_USD_6_Hourly_{current_6_hourly_all_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_6_hourly_all_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                               index=False)
    current_4_hourly_all_market_data_df.to_csv(f"{current_4_hourly_all_market_data_fp}All_Market_Data_BTC_USD_4_Hourly_{current_4_hourly_all_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_4_hourly_all_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                               index=False)
    current_2_hourly_all_market_data_df.to_csv(f"{current_2_hourly_all_market_data_fp}All_Market_Data_BTC_USD_2_Hourly_{current_2_hourly_all_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_2_hourly_all_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                               index=False)
    current_hourly_all_market_data_df.to_csv(f"{current_hourly_all_market_data_fp}All_Market_Data_BTC_USD_Hourly_{current_hourly_all_market_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_hourly_all_market_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                             index=False)

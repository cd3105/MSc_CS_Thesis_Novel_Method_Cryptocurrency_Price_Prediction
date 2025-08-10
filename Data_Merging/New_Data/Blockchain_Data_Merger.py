import pandas as pd
import os
from datetime import datetime

selected_cryptos = ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']

bitinfocharts_path = "All_Crypto_Data/Blockchain_Data/Merged/BitInfoCharts/"
blockchain_btc_path = "All_Crypto_Data/Blockchain_Data/Merged/Blockchain/BTC/1_Day/Blockchain_BTC_USD_Daily_Blockchain_Data_03_01_2009__16_06_2025.csv"
blockchain_via_quandl_btc_path = "All_Crypto_Data/Blockchain_Data/Merged/Blockchain_via_Quandl/BTC/1_Day/Blockchain_via_Quandl_BTC_Daily_Blockchain_Data_02_01_2009__17_06_2025.csv"

for cc in selected_cryptos:
    current_bic_path = f'{bitinfocharts_path}{cc}/1_Day/'
    current_bic_file = os.listdir(current_bic_path)[0]
    current_bic_df = pd.read_csv(f'{current_bic_path}{current_bic_file}')
    current_bic_df['TIMESTAMP'] = pd.to_datetime(current_bic_df['TIMESTAMP'])
    current_only_bic_path = f"All_Crypto_Data/New_Data/Unprocessed/Blockchain_Data/Only_BitInfoCharts_Blockchain_Data/{cc}/1_Day/"

    columns_to_rename = [c for c in current_bic_df.columns[1:]]
    renamed_columns = [f"{c}_PER_BIC" for c in columns_to_rename]
    current_bic_df = current_bic_df.rename(columns=dict(zip(columns_to_rename, renamed_columns)))
    
    if not os.path.exists(current_only_bic_path):
        os.makedirs(current_only_bic_path)
    
    current_bic_df.to_csv(f"{current_only_bic_path}BitInfoCharts_{cc}_Blockchain_Data_{current_bic_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_bic_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                          index=False)


for cc in selected_cryptos:
    current_bic_path = f'{bitinfocharts_path}{cc}/1_Day/'
    current_bic_file = os.listdir(current_bic_path)[0]

    if cc == 'BTC':
        current_daily_bic_df = pd.read_csv(f'{current_bic_path}{current_bic_file}')[['TIMESTAMP', 
                                                                                     'BTC_ACTIVE_ADDRESS_COUNT', 
                                                                                     'BTC_AVERAGE_BLOCK_TIME_MINUTES', 
                                                                                     'BTC_AVERAGE_FEE_PERCENTAGE_IN_TOTAL_BLOCK_REWARD', 
                                                                                     'BTC_AVERAGE_TRANSACTION_FEE_USD', 
                                                                                     'BTC_AVERAGE_TRANSACTION_VALUE_USD', 
                                                                                     'BTC_MEDIAN_TRANSACTION_FEE_USD', 
                                                                                     'BTC_MEDIAN_TRANSACTION_VALUE_USD',
                                                                                     'BTC_MINING_PROFITABILITY',
                                                                                     'BTC_SENT_COINS_USD',
                                                                                     'BTC_SENT_FROM_UNIQUE_ADDRESS_COUNT',
                                                                                     'BTC_TOP_100_RICHEST_PERCENTAGE_OF_TOTAL_COINS']]
        current_daily_blockchain_df = pd.read_csv("All_Crypto_Data/Blockchain_Data/Merged/Blockchain/BTC/1_Day/Blockchain_BTC_USD_Daily_Blockchain_Data_03_01_2009__16_06_2025.csv")[['TIMESTAMP', 
                                                                                                                                                                                      'BTC_MEMPOOL_SIZE_BYTES',
                                                                                                                                                                                      'BTC_MEMPOOL_SIZE_GROWTH_BYTES_PER_SECOND',
                                                                                                                                                                                      'BTC_AVERAGE_TRANSACTION_CONFIRMATION_TIME_MINUTES',
                                                                                                                                                                                      'BTC_MEDIAN_TRANSACTION_CONFIRMATION_TIME_MINUTES',
                                                                                                                                                                                      'BTC_AVERAGE_TRANSACTION_COUNT_PER_BLOCK',
                                                                                                                                                                                      'BTC_AVERAGE_BLOCK_SIZE_MEGABYTES',
                                                                                                                                                                                      'BTC_BLOCKCHAIN_SIZE_MEGABYTES',
                                                                                                                                                                                      'BTC_COST_PER_TRANSACTION_USD',
                                                                                                                                                                                      'BTC_COST_PERCENTAGE_OF_TRADE_VOLUME',
                                                                                                                                                                                      'BTC_TOTAL_ESTIMATED_TRANSACTION_VALUE_BTC',
                                                                                                                                                                                      'BTC_TOTAL_TRANSACTION_FEES_USD',
                                                                                                                                                                                      'BTC_TOTAL_OUTPUT_VALUE_BTC',
                                                                                                                                                                                      'BTC_TOTAL_TRANSACTION_COUNT_PER_SECOND',
                                                                                                                                                                                      'BTC_TOTAL_UNCONFIRMED_TRANSACTION_COUNT_IN_MEMPOOL',
                                                                                                                                                                                      'BTC_TOTAL_UNSPENT_TRANSACTION_OUTPUT_COUNT']]
        current_daily_blockchain_via_quandl_df = pd.read_csv("All_Crypto_Data/Blockchain_Data/Merged/Blockchain_via_Quandl/BTC/1_Day/Blockchain_via_Quandl_BTC_Daily_Blockchain_Data_02_01_2009__17_06_2025.csv").drop(columns=["BTC_MEDIAN_TRANSACTION_CONFIRMATION_TIME_MINUTES",
                                                                                                                                                                                                                                "BTC_TRANSACTION_COUNT_PER_BLOCK",
                                                                                                                                                                                                                                "BTC_AVERAGE_BLOCK_SIZE_MEGABYTES",
                                                                                                                                                                                                                                "BTC_BLOCKCHAIN_SIZE_MEGABYTES",
                                                                                                                                                                                                                                "BTC_COST_PER_TRANSACTION_USD",
                                                                                                                                                                                                                                "BTC_ESTIMATED_TRANSACTION_VOLUME_BTC",
                                                                                                                                                                                                                                "BTC_COST_PERCENTAGE_TRANSACTION_VOLUME",
                                                                                                                                                                                                                                "BTC_TOTAL_TRANSACTION_FEES_USD",
                                                                                                                                                                                                                                "BTC_TOTAL_OUTPUT_VOLUME_BTC",
                                                                                                                                                                                                                                "BTC_TOTAL_TRANSACTION_FEES_BTC",
                                                                                                                                                                                                                                "BTC_TOTAL_TRANSACTION_FEES_BTC",])
        current_daily_blockchain_df['BTC_AVERAGE_BLOCK_SIZE_MEGABYTES'] = current_daily_blockchain_df['BTC_AVERAGE_BLOCK_SIZE_MEGABYTES'] * 10**6
        current_daily_blockchain_df['BTC_BLOCKCHAIN_SIZE_MEGABYTES'] = current_daily_blockchain_df['BTC_BLOCKCHAIN_SIZE_MEGABYTES'] * 10**6
        current_daily_blockchain_df.rename(columns={'BTC_AVERAGE_BLOCK_SIZE_MEGABYTES':'BTC_AVERAGE_BLOCK_SIZE_BYTES',
                                                    'BTC_BLOCKCHAIN_SIZE_MEGABYTES':'BTC_BLOCKCHAIN_SIZE_BYTES'})
        
        blockhain_columns_to_rename = [c for c in current_daily_blockchain_df.columns[1:]]
        blockchain_renamed_columns = [f"{c}_PER_BLOCKCHAIN" for c in blockhain_columns_to_rename]
        current_daily_blockchain_df = current_daily_blockchain_df.rename(columns=dict(zip(blockhain_columns_to_rename, 
                                                                                          blockchain_renamed_columns)))
        

        current_daily_blockchain_via_quandl_df["BTC_HASH_RATE_TERAHASHES_PER_SECOND"] = current_daily_blockchain_via_quandl_df["BTC_HASH_RATE_TERAHASHES_PER_SECOND"] * 10**12
        current_daily_blockchain_via_quandl_df.rename(columns={'BTC_HASH_RATE_TERAHASHES_PER_SECOND':'BTC_HASH_RATE_HASHES_PER_SECOND',})

        blockhain_via_quandl_columns_to_rename = [c for c in current_daily_blockchain_via_quandl_df.columns[1:]]
        blockchain_via_quandl_renamed_columns = [f"{c}_PER_BLOCKCHAIN" for c in blockhain_via_quandl_columns_to_rename]
        current_daily_blockchain_via_quandl_df = current_daily_blockchain_via_quandl_df.rename(columns=dict(zip(blockhain_via_quandl_columns_to_rename, 
                                                                                                                blockchain_via_quandl_renamed_columns)))
        
        
        bic_columns_to_rename = [c for c in current_daily_bic_df.columns[1:]]
        bic_renamed_columns = [f"{c}_PER_BIC" for c in bic_columns_to_rename]
        current_daily_bic_df = current_daily_bic_df.rename(columns=dict(zip(bic_columns_to_rename, 
                                                                            bic_renamed_columns)))

        current_daily_bic_df['TIMESTAMP'] = pd.to_datetime(current_daily_bic_df['TIMESTAMP'])
        current_daily_blockchain_df['TIMESTAMP'] = pd.to_datetime(current_daily_blockchain_df['TIMESTAMP'])
        current_daily_blockchain_via_quandl_df['TIMESTAMP'] = pd.to_datetime(current_daily_blockchain_via_quandl_df['TIMESTAMP'])


        current_daily_all_blockchain_df = pd.merge(current_daily_blockchain_via_quandl_df,
                                                   pd.merge(current_daily_blockchain_df, 
                                                            current_daily_bic_df,
                                                            on="TIMESTAMP",
                                                            how="left"),
                                                   on="TIMESTAMP",
                                                   how="left")


        btc_12_hourly_all_blockchain_data_df = pd.read_csv("All_Crypto_Data/Blockchain_Data/Merged/Blockchain/BTC/12_Hour/Blockchain_BTC_USD_12_Hourly_Blockchain_Data_17_01_2009__16_06_2025.csv")
        btc_8_hourly_all_blockchain_data_df = pd.read_csv("All_Crypto_Data/Blockchain_Data/Merged/Blockchain/BTC/8_Hour/Blockchain_BTC_USD_8_Hourly_Blockchain_Data_17_01_2009__16_06_2025.csv")
        btc_6_hourly_all_blockchain_data_df = pd.read_csv("All_Crypto_Data/Blockchain_Data/Merged/Blockchain/BTC/6_Hour/Blockchain_BTC_USD_6_Hourly_Blockchain_Data_17_01_2009__16_06_2025.csv")
        btc_4_hourly_all_blockchain_data_df = pd.read_csv("All_Crypto_Data/Blockchain_Data/Merged/Blockchain/BTC/4_Hour/Blockchain_BTC_USD_4_Hourly_Blockchain_Data_17_01_2009__16_06_2025.csv")
        btc_2_hourly_all_blockchain_data_df = pd.read_csv("All_Crypto_Data/Blockchain_Data/Merged/Blockchain/BTC/2_Hour/Blockchain_BTC_USD_2_Hourly_Blockchain_Data_17_01_2009__16_06_2025.csv")
        btc_1_hourly_all_blockchain_data_df = pd.read_csv("All_Crypto_Data/Blockchain_Data/Merged/Blockchain/BTC/1_Hour/Blockchain_BTC_USD_Hourly_Blockchain_Data_17_01_2009__16_06_2025.csv")

        btc_12_hourly_all_blockchain_data_df['TIMESTAMP'] = pd.to_datetime(btc_12_hourly_all_blockchain_data_df['TIMESTAMP'])
        btc_8_hourly_all_blockchain_data_df['TIMESTAMP'] = pd.to_datetime(btc_8_hourly_all_blockchain_data_df['TIMESTAMP'])
        btc_6_hourly_all_blockchain_data_df['TIMESTAMP'] = pd.to_datetime(btc_6_hourly_all_blockchain_data_df['TIMESTAMP'])
        btc_4_hourly_all_blockchain_data_df['TIMESTAMP'] = pd.to_datetime(btc_4_hourly_all_blockchain_data_df['TIMESTAMP'])
        btc_2_hourly_all_blockchain_data_df['TIMESTAMP'] = pd.to_datetime(btc_2_hourly_all_blockchain_data_df['TIMESTAMP'])
        btc_1_hourly_all_blockchain_data_df['TIMESTAMP'] = pd.to_datetime(btc_1_hourly_all_blockchain_data_df['TIMESTAMP'])

        btc_12_hourly_all_blockchain_data_path = "All_Crypto_Data/New_Data/Unprocessed/Blockchain_Data/All_Blockchain_Data/BTC/12_Hour/"
        btc_8_hourly_all_blockchain_data_path = "All_Crypto_Data/New_Data/Unprocessed/Blockchain_Data/All_Blockchain_Data/BTC/8_Hour/"
        btc_6_hourly_all_blockchain_data_path = "All_Crypto_Data/New_Data/Unprocessed/Blockchain_Data/All_Blockchain_Data/BTC/6_Hour/"
        btc_4_hourly_all_blockchain_data_path = "All_Crypto_Data/New_Data/Unprocessed/Blockchain_Data/All_Blockchain_Data/BTC/4_Hour/"
        btc_2_hourly_all_blockchain_data_path = "All_Crypto_Data/New_Data/Unprocessed/Blockchain_Data/All_Blockchain_Data/BTC/2_Hour/"
        btc_1_hourly_all_blockchain_data_path = "All_Crypto_Data/New_Data/Unprocessed/Blockchain_Data/All_Blockchain_Data/BTC/1_Hour/"


        if not os.path.exists(btc_12_hourly_all_blockchain_data_path):
            os.makedirs(btc_12_hourly_all_blockchain_data_path)
        
        if not os.path.exists(btc_8_hourly_all_blockchain_data_path):
            os.makedirs(btc_8_hourly_all_blockchain_data_path)
        
        if not os.path.exists(btc_6_hourly_all_blockchain_data_path):
            os.makedirs(btc_6_hourly_all_blockchain_data_path)

        if not os.path.exists(btc_4_hourly_all_blockchain_data_path):
            os.makedirs(btc_4_hourly_all_blockchain_data_path)

        if not os.path.exists(btc_2_hourly_all_blockchain_data_path):
            os.makedirs(btc_2_hourly_all_blockchain_data_path)

        if not os.path.exists(btc_1_hourly_all_blockchain_data_path):
            os.makedirs(btc_1_hourly_all_blockchain_data_path)


        btc_12_hourly_all_blockchain_data_df.to_csv(f"{btc_12_hourly_all_blockchain_data_path}All_Blockchain_Data_{cc}_USD_{btc_12_hourly_all_blockchain_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{btc_12_hourly_all_blockchain_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                                    index=False)
        btc_8_hourly_all_blockchain_data_df.to_csv(f"{btc_8_hourly_all_blockchain_data_path}All_Blockchain_Data_{cc}_USD_{btc_8_hourly_all_blockchain_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{btc_8_hourly_all_blockchain_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                                    index=False)
        btc_6_hourly_all_blockchain_data_df.to_csv(f"{btc_6_hourly_all_blockchain_data_path}All_Blockchain_Data_{cc}_USD_{btc_6_hourly_all_blockchain_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{btc_6_hourly_all_blockchain_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                                    index=False)
        btc_4_hourly_all_blockchain_data_df.to_csv(f"{btc_4_hourly_all_blockchain_data_path}All_Blockchain_Data_{cc}_USD_{btc_4_hourly_all_blockchain_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{btc_4_hourly_all_blockchain_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                                    index=False)
        btc_2_hourly_all_blockchain_data_df.to_csv(f"{btc_2_hourly_all_blockchain_data_path}All_Blockchain_Data_{cc}_USD_{btc_2_hourly_all_blockchain_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{btc_2_hourly_all_blockchain_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                                    index=False)
        btc_1_hourly_all_blockchain_data_df.to_csv(f"{btc_1_hourly_all_blockchain_data_path}All_Blockchain_Data_{cc}_USD_{btc_1_hourly_all_blockchain_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{btc_1_hourly_all_blockchain_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                                    index=False)
    elif cc == 'ETH':
        current_daily_bic_df = pd.read_csv(f'{current_bic_path}{current_bic_file}')[['TIMESTAMP', 
                                                                                     'ETH_MINING_PROFITABILITY',
                                                                                     'ETH_MEDIAN_TRANSACTION_FEE_USD',
                                                                                     'ETH_AVERAGE_TRANSACTION_VALUE_USD',
                                                                                     'ETH_MEDIAN_TRANSACTION_VALUE_USD',
                                                                                     'ETH_SENT_COINS_USD',
                                                                                     'ETH_AVERAGE_FEE_PERCENTAGE_IN_TOTAL_BLOCK_REWARD']]
        current_daily_etherscan_df = pd.read_csv("All_Crypto_Data/Blockchain_Data/Merged/Etherscan/ETH/1_Day/Etherscan_ETH_Daily_UTC_Blockchain_Data_30_07_2015__14_05_2025.csv")

        current_daily_bic_df['TIMESTAMP'] = pd.to_datetime(current_daily_bic_df['TIMESTAMP'])
        current_daily_etherscan_df['TIMESTAMP'] = pd.to_datetime(current_daily_etherscan_df['TIMESTAMP'])

        current_daily_etherscan_df['ETH_NETWORK_HASH_RATE_GIGAHASHES_PER_SECOND'] = current_daily_etherscan_df['ETH_NETWORK_HASH_RATE_GIGAHASHES_PER_SECOND'] * 10**9
        current_daily_etherscan_df['ETH_NETWORK_DIFFICULTY_TERAHASHES'] = current_daily_etherscan_df['ETH_NETWORK_DIFFICULTY_TERAHASHES'] * 10**12


        current_daily_etherscan_df.rename(columns={'ETH_NETWORK_HASH_RATE_GIGAHASHES_PER_SECOND': 'ETH_NETWORK_HASH_RATE_HASHES_PER_SECOND',
                                                   'ETH_NETWORK_DIFFICULTY_TERAHASHES': 'ETH_NETWORK_DIFFICULTY_HASHES',})
        
        etherscan_columns_to_rename = [c for c in current_daily_etherscan_df.columns[1:]]
        etherscan_renamed_columns = [f"{c}_PER_ETHERSCAN" for c in etherscan_columns_to_rename]
        current_daily_etherscan_df = current_daily_etherscan_df.rename(columns=dict(zip(etherscan_columns_to_rename, 
                                                                                        etherscan_renamed_columns)))
        
        bic_columns_to_rename = [c for c in current_daily_bic_df.columns[1:]]
        bic_renamed_columns = [f"{c}_PER_BIC" for c in bic_columns_to_rename]
        current_daily_bic_df = current_daily_bic_df.rename(columns=dict(zip(bic_columns_to_rename, 
                                                                            bic_renamed_columns)))
        
        
        current_daily_all_blockchain_df = pd.merge(current_daily_etherscan_df,
                                                   current_daily_bic_df,
                                                   on="TIMESTAMP",
                                                   how="left")
    else:
        current_daily_all_blockchain_df = pd.read_csv(f'{current_bic_path}{current_bic_file}')
        current_daily_all_blockchain_df['TIMESTAMP'] = pd.to_datetime(current_daily_all_blockchain_df['TIMESTAMP'])
        current_daily_all_blockchain_path = f"All_Crypto_Data/New_Data/Unprocessed/Blockchain_Data/All_Blockchain_Data/{cc}/1_Day/"

        columns_to_rename = [c for c in current_daily_all_blockchain_df.columns[1:]]
        renamed_columns = [f"{c}_PER_BIC" for c in columns_to_rename]
        current_daily_all_blockchain_df = current_daily_all_blockchain_df.rename(columns=dict(zip(columns_to_rename, 
                                                                                                  renamed_columns)))
    
    current_daily_all_blockchain_path = f"All_Crypto_Data/New_Data/Unprocessed/Blockchain_Data/All_Blockchain_Data/{cc}/1_Day/"

    if not os.path.exists(current_daily_all_blockchain_path):
        os.makedirs(current_daily_all_blockchain_path)
        
    current_daily_all_blockchain_df.to_csv(f"{current_daily_all_blockchain_path}All_Blockchain_Data_{cc}_USD_{current_daily_all_blockchain_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_daily_all_blockchain_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                           index=False)


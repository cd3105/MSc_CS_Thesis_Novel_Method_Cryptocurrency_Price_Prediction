import pandas as pd
from datetime import datetime
import os

# Script for splitting Market Data and Blockchain Data into separate datasets

base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/CryptoCompare/"

for cc in [m for m in os.listdir(base_path) if not m.endswith('.txt')]:
    file_path = f'{base_path}{cc}/1_Day/'
    file_name = os.listdir(file_path)[0]

    if cc in ['BTC', 'ETH', 'LTC']:
        cc_df = pd.read_csv(f'{file_path}{file_name}').drop(['timeDate', 'ts', 'id', 'symbol'], axis=1).rename(columns={'time':'TIMESTAMP',
                                                                                                                        'close':f'{cc}_CLOSE_PRICE_USD',
                                                                                                                        'high':f'{cc}_HIGH_PRICE_USD',
                                                                                                                        'low':f'{cc}_LOW_PRICE_USD',
                                                                                                                        'open':f'{cc}_OPEN_PRICE_USD',
                                                                                                                        'volumefrom':f'{cc}_VOLUME_FROM_USD',
                                                                                                                        'volumeto':f'{cc}_VOLUME_TO_USD',
                                                                                                                        'zero_balance_addresses_all_time':f'{cc}_ALL_TIME_ZERO_BALANCE_ADDRESS_COUNT',
                                                                                                                        'unique_addresses_all_time':f'{cc}_ALL_TIME_UNIQUE_ADDRESS_COUNT',
                                                                                                                        'new_addresses':f'{cc}_NEW_ADDRESS_COUNT',
                                                                                                                        'active_addresses':f'{cc}_ACTIVE_ADDRESS_COUNT',
                                                                                                                        'transaction_count':f'{cc}_TRANSACTION_COUNT',
                                                                                                                        'transaction_count_all_time':f'{cc}_ALL_TIME_TRANSACTION_COUNT',
                                                                                                                        'large_transaction_count':f'{cc}_LARGE_TRANSACTION_COUNT',
                                                                                                                        'average_transaction_value':f'{cc}_AVERAGE_TRANSACTION_VALUE',
                                                                                                                        'block_height':f'{cc}_BLOCK_HEIGHT',
                                                                                                                        'hashrate':f'{cc}_HASH_RATE',
                                                                                                                        'difficulty':f'{cc}_DIFFICULTY',
                                                                                                                        'block_time':f'{cc}_BLOCK_TIME',
                                                                                                                        'block_size':f'{cc}_BLOCK_SIZE',
                                                                                                                        'current_supply':f'{cc}_CURRENT_SUPPLY',})
        cc_df['TIMESTAMP'] = cc_df['TIMESTAMP'].apply(lambda x: datetime.utcfromtimestamp((x/1000)) if isinstance(x, int) else datetime.utcfromtimestamp(x.timestamp()))
        blockchain_path = f'All_Crypto_Data/Blockchain_Data/Merged/CryptoCompare/{cc}/1_Day/'

        if not os.path.exists(blockchain_path):
            os.makedirs(blockchain_path)
        
        cc_df[['TIMESTAMP', 
               f'{cc}_ALL_TIME_ZERO_BALANCE_ADDRESS_COUNT', 
               f'{cc}_ALL_TIME_UNIQUE_ADDRESS_COUNT', 
               f'{cc}_NEW_ADDRESS_COUNT', 
               f'{cc}_ACTIVE_ADDRESS_COUNT', 
               f'{cc}_TRANSACTION_COUNT',
               f'{cc}_ALL_TIME_TRANSACTION_COUNT',
               f'{cc}_LARGE_TRANSACTION_COUNT',
               f'{cc}_AVERAGE_TRANSACTION_VALUE',
               f'{cc}_BLOCK_HEIGHT',
               f'{cc}_HASH_RATE',
               f'{cc}_DIFFICULTY',
               f'{cc}_BLOCK_TIME',
               f'{cc}_BLOCK_SIZE',
               f'{cc}_CURRENT_SUPPLY',]].to_csv(f"{blockchain_path}CryptoCompare_{cc}_Daily_Blockchain_Data_{cc_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{cc_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", index=False)
        

    else:
        cc_df = pd.read_csv(f'{file_path}{file_name}').drop(['timeDate', 'ts'], axis=1).rename(columns={'time':'TIMESTAMP',
                                                                                                        'close':f'{cc}_CLOSE_PRICE_USD',
                                                                                                        'high':f'{cc}_HIGH_PRICE_USD',
                                                                                                        'low':f'{cc}_LOW_PRICE_USD',
                                                                                                        'open':f'{cc}_OPEN_PRICE_USD',
                                                                                                        'volumefrom':f'{cc}_VOLUME_FROM_USD',
                                                                                                        'volumeto':f'{cc}_VOLUME_TO_USD',})
        cc_df['TIMESTAMP'] = cc_df['TIMESTAMP'].apply(lambda x: datetime.utcfromtimestamp((x/1000)) if isinstance(x, int) else datetime.utcfromtimestamp(x.timestamp()))
    
    cc_df[['TIMESTAMP', 
           f'{cc}_CLOSE_PRICE_USD', 
           f'{cc}_HIGH_PRICE_USD', 
           f'{cc}_LOW_PRICE_USD',
           f'{cc}_OPEN_PRICE_USD',
           f'{cc}_VOLUME_FROM_USD',
           f'{cc}_VOLUME_TO_USD',]].to_csv(f"{file_path}CryptoCompare_{cc}_Daily_Market_Data_{cc_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{cc_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", index=False)

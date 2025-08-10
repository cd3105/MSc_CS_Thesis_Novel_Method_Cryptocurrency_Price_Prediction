import pandas as pd
import os
import re

base_file_path = "All_Crypto_Data/Blockchain_Data/Unmerged/Etherscan/ETH/1_Day/"
merged_df = pd.DataFrame({'TIMESTAMP':[]})

for csv in os.listdir(base_file_path):
    current_df = pd.read_csv(f"{base_file_path}{csv}").rename(columns={'Date(UTC)':'TIMESTAMP'})
    current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])

    if 'UnixTimeStamp' in current_df.columns:
        current_df = current_df.drop(['UnixTimeStamp'], axis=1)

    if 'DateTime' in current_df.columns:
        current_df = current_df.drop(['DateTime'], axis=1)

    feature = re.search(r'UTC_(.*?)_(?=\d)', csv).group(1)

    if (len(current_df.columns) == 2) and ('Value' in str(current_df.columns[1])):
        current_df = current_df.rename(columns={current_df.columns[1]:f'ETH_{feature.upper()}'})
    elif csv == 'Etherscan_ETH_USD_Daily_UTC_Average_Transaction_Fee_30_07_2015__14_05_2025.csv':
        original_columns = current_df.columns[1:]
        new_columns = [f'ETH_{"_".join(oc.split(" ")).upper().replace("(", "").replace(")", "")}' for oc in original_columns]

        current_df = current_df.rename(columns=dict(zip(original_columns, new_columns)))
    elif len(current_df.columns) > 2:
        original_columns = current_df.columns[1:]
        new_columns = [f'ETH_{feature.upper()}_{"_".join(oc.split(" ")).upper()}' for oc in original_columns]

        current_df = current_df.rename(columns=dict(zip(original_columns, new_columns)))
    elif csv == 'Etherscan_ETH_USD_Daily_UTC_ETH_Burnt_06_08_2021__14_05_2025.csv':
        current_df = current_df.rename(columns={'BurntFees':'ETH_BURNT_FEES_ETH'})
    else:
        current_df = current_df.rename(columns={'No. of ERC20 Token Transfers':'ETH_ERC20_TOKEN_TRANSFER_COUNT'})
    
    merged_df = pd.merge(merged_df, current_df, on='TIMESTAMP', how='outer')

merged_file_path = "All_Crypto_Data/Blockchain_Data/Merged/Etherscan/ETH/1_Day/"

if not os.path.exists(merged_file_path):
    os.makedirs(merged_file_path)

merged_df.to_csv(f"{merged_file_path}Etherscan_ETH_Daily_UTC_Blockchain_Data_{merged_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                 index=False)

import pandas as pd

# Script for adding Commas to retrieved Etherscan Network Transaction Fees

df = pd.read_csv(r"All_Crypto_Data/Blockchain_Data/Unmerged/Etherscan/ETH/1_Day/Etherscan_ETH_USD_Daily_UTC_Network_Transaction_Fee_ETH_30_07_2015__14_05_2025.csv")

df['Length'] = df['Value'].apply(lambda x: len(str(x)))
df['Value'] = df.apply(lambda x: float(f"{str(x.Value)[:(x.Length-18)]}.{str(x.Value)[(x.Length-18):]}") if x.Length > 1 else 0, axis=1)
df = df.drop(['Length'], axis=1)

df.to_csv(r"All_Crypto_Data/Blockchain_Data/Unmerged/Etherscan/ETH/1_Day/Etherscan_ETH_USD_Daily_UTC_Network_Transaction_Fee_ETH_30_07_2015__14_05_2025.csv", index=False)

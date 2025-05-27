import nasdaqdatalink
import pandas as pd

data_code_dict = {"TOTBC": "Total_Coins",
                  "ATRCT": "Median_Transaction_Confirmation_Time",
                  "AVBLS": "Average_Block_Size",
                  "BLCHS": "API_Blockchain_Size",
                  "CPTRA": "Cost_Per_Transaction",
                  "CPTRV": "Cost_Percentage_Transaction_Volume",
                  "DIFF": "Difficulty",
                  "ETRAV": "Estimated_Transaction_Volume",
                  "ETRVU": "Estimated_Transaction_Volume_USD",
                  "HRATE": "Hash_Rate",
                  "MIREV": "Miners_Revenue",
                  "MWNTD": "My_Wallet_Transaction_Count",
                  "MWNUS": "My_Wallet_User_Count",
                  "MWTRV": "My_Wallet_Transaction_Volume",
                  "NADDU": "Unique_Addresses_Used_Count",
                  "NTRAN": "Transaction_Count",
                  "NTRAT": "Total_Transaction_Count",
                  "NTRBL": "Transaction_Count_Per_Block",
                  "NTREP": "Transaction_Count_Excluding_Popular_Addresses",
                  "TOUTV": "Total_Output_Volume",
                  "TRFEE": "Total_Transaction_Fees",
                  "TRFUS": "Total_Transaction_Fees_USD",
                  "BCDDC": "Cumulative_Days_Destroyed",
                  "BCDDE": "Days_Destroyed",
                  "BCDDM": "Minimum_Age_1_Month_Days_Destroyed",
                  "BCDDW": "Minimum_Age_1_Week_Days_Destroyed",
                  "BCDDY": "Minimum_Age_1_Year_Days_Destroyed",
                  "NETDF": "Network_Deficit",
                  "TVTVR": "Trade_Volume_VS_Transaction_Volume_Ratio"}

nasdaqdatalink.ApiConfig.api_key = "EMdb_Jf-aEGJU3xsUZEi"
full_df = pd.DataFrame(columns=['Time'])
file_path = "All_Crypto_Data/Blockchain_Data/Blockchain_via_Quandl/BTC/1_Day/"

for k in data_code_dict.keys():
    current_df = nasdaqdatalink.get_table('QDL/BCHAIN',code=k).drop(['code'], axis=1).rename(columns={'date':'Time', 'value': data_code_dict[k]}).sort_values(by='Time').reset_index(drop=True)
    current_file_name = f"Blockchain_via_Quandl_BTC_USD_Daily_{data_code_dict[k]}_{list(current_df['Time'])[0].strftime('%d-%m-%Y').replace('-', '_')}__{list(current_df['Time'])[-1].strftime('%d-%m-%Y').replace('-', '_')}.csv"
    full_df = pd.merge(full_df, current_df, on='Time', how='outer')

    current_df.to_csv(file_path + current_file_name)

full_df.to_csv(file_path + f"ALL_Blockchain_via_Quandl_BTC_USD_Daily_Blockchain_Data_{list(full_df['Time'])[0].strftime('%d-%m-%Y').replace('-', '_')}__{list(full_df['Time'])[-1].strftime('%d-%m-%Y').replace('-', '_')}.csv")

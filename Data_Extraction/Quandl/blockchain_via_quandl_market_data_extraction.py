import nasdaqdatalink
import pandas as pd

data_code_dict = {"MKPRU": "Market_Price_USD",
                  "MKTCP": "Market_Cap",
                  "TRVOU": "USD_Exchange_Trade_Volume",}

nasdaqdatalink.ApiConfig.api_key = "EMdb_Jf-aEGJU3xsUZEi"
full_df = pd.DataFrame(columns=['Time'])
file_path = "All_Crypto_Data/Crypto_Market_Data/Blockchain_via_Quandl/BTC/1_Day/"

for k in data_code_dict.keys():
    current_df = nasdaqdatalink.get_table('QDL/BCHAIN',code=k).drop(['code'], axis=1).rename(columns={'date':'Time', 'value': data_code_dict[k]}).sort_values(by='Time').reset_index(drop=True)
    current_file_name = f"Blockchain_via_Quandl_BTC_USD_Daily_{data_code_dict[k]}_{list(current_df['Time'])[0].strftime('%d-%m-%Y').replace('-', '_')}__{list(current_df['Time'])[-1].strftime('%d-%m-%Y').replace('-', '_')}.csv"
    full_df = pd.merge(full_df, current_df, on='Time', how='outer')

    current_df.to_csv(file_path + current_file_name)

full_df.to_csv(file_path + f"ALL_Blockchain_via_Quandl_BTC_USD_Daily_Market_Data_{list(full_df['Time'])[0].strftime('%d-%m-%Y').replace('-', '_')}__{list(full_df['Time'])[-1].strftime('%d-%m-%Y').replace('-', '_')}.csv")

import pandas as pd
import os
from datetime import datetime

# Script to Merge DataHub Data into a single Dataset

daily_base_path = "All_Crypto_Data/Macroeconomic_Data/Unmerged/DataHub/1_Day/"
monthly_base_path = "All_Crypto_Data/Macroeconomic_Data/Unmerged/DataHub/1_Month/"

brent_oil_price_df = pd.read_csv(f"{daily_base_path}DataHub_Daily_Brent_Oil_Price_USD_20_05_1987__12_05_2025.csv").rename(columns={'Date':'TIMESTAMP',
                                                                                                                                   'Price':'BRENT_OIL_PRICE_USD'})
natural_gas_price_df = pd.read_csv(f"{daily_base_path}DataHub_Daily_Natural_Gas_Price_USD_07_01_1997__12_05_2025.csv").rename(columns={'Date':'TIMESTAMP',
                                                                                                                                       'Price':'NATURAL_GAS_PRICE_USD'})
vix_index_df = pd.read_csv(f"{daily_base_path}DataHub_Daily_VIX_Index_USD_01_02_1990__14_05_2025.csv").rename(columns={'DATE':'TIMESTAMP',
                                                                                                                       'OPEN':'VIX_OPEN_PRICE_USD',
                                                                                                                       'HIGH':'VIX_HIGH_PRICE_USD',
                                                                                                                       'LOW':'VIX_LOW_PRICE_USD',
                                                                                                                       'CLOSE':'VIX_CLOSE_PRICE_USD',})
wti_oil_price_df = pd.read_csv(f"{daily_base_path}DataHub_Daily_WTI_Oil_Price_USD_02_01_1986__12_05_2025.csv").rename(columns={'Date':'TIMESTAMP',
                                                                                                                               'Price':'WTI_OIL_PRICE_USD'})
gold_price_df = pd.read_csv(f"{monthly_base_path}DataHub_Monthly_Gold_Price.csv").rename(columns={'Date':'TIMESTAMP',
                                                                                                  'Price':'GOLD_PRICE_USD'})
s_n_p500_price_df = pd.read_csv(f"{monthly_base_path}DataHub_Monthly_S&P500_Index_USD_01_01_1871__01_09_2023.csv").rename(columns={'Date':'TIMESTAMP',
                                                                                                                                   'SP500':'S&P500_INDEX_USD',
                                                                                                                                   'Dividend':'S&P500_DIVIDEND_USD',
                                                                                                                                   'Earnings':'S&P500_EARNINGS_USD',
                                                                                                                                   'Consumer Price Index':'S&P500_CONSUMER_PRICE_INDEX_USD',
                                                                                                                                   'Long Interest Rate':'S&P500_LONG_INTEREST_RATE_USD',
                                                                                                                                   'Real Price':'S&P500_REAL_PRICE_USD',
                                                                                                                                   'Real Dividend':'S&P500_REAL_DIVIDEND_USD',
                                                                                                                                   'Real Earnings':'S&P500_REAL_EARNINGS_USD',
                                                                                                                                   'PE10':'S&P500_PE10_USD'})

brent_oil_price_df['TIMESTAMP'] = pd.to_datetime(brent_oil_price_df['TIMESTAMP'])
natural_gas_price_df['TIMESTAMP'] = pd.to_datetime(natural_gas_price_df['TIMESTAMP'])
vix_index_df['TIMESTAMP'] = pd.to_datetime(vix_index_df['TIMESTAMP'])
wti_oil_price_df['TIMESTAMP'] = pd.to_datetime(wti_oil_price_df['TIMESTAMP'])

gold_price_df['TIMESTAMP'] = pd.to_datetime(gold_price_df['TIMESTAMP'])
s_n_p500_price_df['TIMESTAMP'] = pd.to_datetime(s_n_p500_price_df['TIMESTAMP'])

dfs = [brent_oil_price_df, natural_gas_price_df, vix_index_df, wti_oil_price_df, gold_price_df, s_n_p500_price_df]

for i, df in enumerate(dfs):
    current_date_range_df = pd.DataFrame({
        'TIMESTAMP': pd.date_range(start=datetime(2009, 1, 1), end=df['TIMESTAMP'].max(), freq='D')
    }).sort_values('TIMESTAMP', ascending=True)
    current_df = pd.merge(df, current_date_range_df, on='TIMESTAMP', how='right')
    current_df = current_df[current_df['TIMESTAMP'] >= datetime(2009, 1, 1)]
    dfs[i] = current_df

merged_df = pd.DataFrame({'TIMESTAMP':[]})

for df in dfs:
    merged_df = pd.merge(merged_df, df, on='TIMESTAMP', how='outer').sort_values('TIMESTAMP').reset_index(drop=True)

new_map_path = "All_Crypto_Data/Macroeconomic_Data/Merged/DataHub/1_Day/"
file_name = f"DataHub_Daily_Macroeconomic_Data_USD_{merged_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"

if not os.path.exists(new_map_path):
    os.makedirs(new_map_path)

merged_df.to_csv(f"{new_map_path}{file_name}", index=False)

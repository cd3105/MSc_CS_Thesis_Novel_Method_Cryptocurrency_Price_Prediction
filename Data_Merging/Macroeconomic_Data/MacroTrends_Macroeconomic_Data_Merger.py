import pandas as pd
import os
import re
from datetime import datetime

# Script to Merge MacroTrends Data into a single Dataset

base_path = "All_Crypto_Data/Macroeconomic_Data/Unmerged/MacroTrends/1_Day/"

naming_dict = {'MacroTrends_10_Year_Bond_Yield_USD_Daily_02_01_1962__15_05_2025.csv':'10_YEAR_BOND_YIELD_USD', 
               'MacroTrends_1_Year_Bond_Yield_USD_Daily_02_01_1962__15_05_2025.csv':'1_YEAR_BOND_YIELD_USD', 
               'MacroTrends_30_Year_Bond_Yield_USD_Daily_15_02_1977__15_05_2025.csv':'30_YEAR_BOND_YIELD_USD', 
               'MacroTrends_5_Year_Bond_Yield_USD_Daily_02_01_1962__15_05_2025.csv':'5_YEAR_BOND_YIELD_USD', 
               'MacroTrends_Brent_Oil_Price_USD_Daily_18_05_2015__16_05_2025.csv':'BRENT_OIL_PRICE_USD', 
               'MacroTrends_CAC40_Index_EUR_Daily_01_03_1990__02_05_2025.csv':'CAC40_INDEX_EUR', 
               'MacroTrends_Crude_Oil_Price_USD_Daily_18_05_2015__16_05_2025.csv':'CRUDE_OIL_PRICE_USD', 
               'MacroTrends_DJI_Index_USD_Daily_18_05_2012__16_05_2025.csv':'DJI_INDEX_USD', 
               'MacroTrends_Federal_Funds_Rate_USD_Daily_01_07_1954__15_05_2025.csv':'FEDERAL_FUNDS_RATE_USD', 
               'MacroTrends_Gold_Price_USD_Daily_18_05_2012__16_05_2025.csv':'GOLD_PRICE_USD', 
               'MacroTrends_HSI_Index_HKD_Daily_31_12_1986__02_05_2025.csv':'HSI_INDEX_HKD', 
               'MacroTrends_NASDAQ_Index_USD_Daily_18_05_2012__15_05_2025.csv':'NASDAQ_INDEX_USD', 
               'MacroTrends_Natural_Gas_Price_USD_Daily_18_05_2012__12_05_2025.csv':'NATURAL_GAS_PRICE_USD', 
               'MacroTrends_Nikkei225_Index_JPY_Daily_16_05_1949__16_05_2025.csv':'NIKKEI225_INDEX_JPY', 
               'MacroTrends_Silver_Price_USD_Daily_18_05_2012__16_05_2025.csv':'SILVER_PRICE_USD', 
               'MacroTrends_VIX_Index_USD_Daily_02_01_1990__15_05_2025.csv':'VIX_INDEX_USD'}

merged_df = pd.DataFrame(columns={'TIMESTAMP':[]})

for f in os.listdir(base_path):
    current_df = pd.read_csv(f"{base_path}{f}").dropna().rename(columns={'date':'TIMESTAMP',
                                                                         ' value':re.search(r'MacroTrends_(.*?)_Daily', f).group(1).upper()})
    current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'], format="%d/%m/%Y")
    current_date_range_df = pd.DataFrame({
        'TIMESTAMP': pd.date_range(start=datetime(2009, 1, 1), end=current_df['TIMESTAMP'].max(), freq='D')
    }).sort_values('TIMESTAMP', ascending=True)
    current_df = pd.merge(current_df, current_date_range_df, on='TIMESTAMP', how='right')
    merged_df = pd.merge(merged_df, current_df, on='TIMESTAMP', how='outer').sort_values('TIMESTAMP').reset_index(drop=True)

new_map_path = "All_Crypto_Data/Macroeconomic_Data/Merged/MacroTrends/1_Day/"
file_name = f"MacroTrends_Daily_Macroeconomic_Data_USD_{merged_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{merged_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"

if not os.path.exists(new_map_path):
    os.makedirs(new_map_path)

merged_df.to_csv(f"{new_map_path}{file_name}", index=False)

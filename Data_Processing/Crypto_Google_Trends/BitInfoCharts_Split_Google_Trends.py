import pandas as pd
import os

# Script for splitting Google Trends data of each cryptocurrency into separate datasets

gt_df = pd.read_csv("All_Crypto_Data/Crypto_Google_Trends/Merged/Google_Trends_via_BitInfoCharts/1_Day/BitInfoCharts_Daily_Google_Trends_Full_Name_BTC_ETH_DOGE_LTC_XMR_01_01_2010__18_05_2025.csv", 
                    index_col=0).rename(columns={'Time':'TIMESTAMP'})
gt_df['TIMESTAMP'] = pd.to_datetime(gt_df['TIMESTAMP'])

for c in gt_df.columns[1:]:
    current_cc = c.split('_')[0]
    current_subset = gt_df[['TIMESTAMP', c]].rename(columns={c:c.upper()})

    current_map_name = f"All_Crypto_Data/Crypto_Google_Trends/Merged/Google_Trends_via_BitInfoCharts/{current_cc}/1_Day/"
    current_file_name = f"BitInfoCharts_Daily_Google_Trends_Full_Name_{c}_{current_subset['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_subset['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"

    if not os.path.exists(current_map_name):
        os.makedirs(current_map_name)
        
    current_subset.to_csv(f"{current_map_name}{current_file_name}", index=False)
    
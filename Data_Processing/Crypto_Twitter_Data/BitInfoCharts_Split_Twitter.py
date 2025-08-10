import pandas as pd
import os
import re

base_path = 'All_Crypto_Data/Crypto_Twitter_Data/BitInfoCharts/1_Day/'
merged_df = pd.DataFrame({'TIMESTAMP':[]})

for csv in os.listdir(base_path):
    current_df = pd.read_csv(f"{base_path}{csv}", index_col=0).rename(columns={'Time':'TIMESTAMP'})
    current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])
    
    merged_df = pd.merge(merged_df, current_df, on='TIMESTAMP', how='outer').sort_values('TIMESTAMP')

merged_df = merged_df.rename(columns=dict(zip(merged_df.columns, [re.sub(r'_\d$', '', c).upper() for c in merged_df.columns])))
ccs = list(set([c.split('_')[0] for c in merged_df.columns[1:]]))

for cc in ccs:
    current_column_subset = ['TIMESTAMP'] + [c for c in merged_df.columns[1:] if cc in c]
    current_cc_df = merged_df[current_column_subset]
    current_cc_path = f"All_Crypto_Data/Crypto_Twitter_Data/BitInfoCharts/{cc}/1_Day/"
    
    if not os.path.exists(current_cc_path):
        os.makedirs(current_cc_path)
    
    current_cc_df.to_csv(f"{current_cc_path}BitInfoCharts_{cc}_Daily_Tweet_Volume_Data.csv", index=False)

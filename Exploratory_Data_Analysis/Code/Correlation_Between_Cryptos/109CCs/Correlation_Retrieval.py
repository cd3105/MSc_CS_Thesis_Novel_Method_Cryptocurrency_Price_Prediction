import os 
import pandas as pd 
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import grangercausalitytests

base_data_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Kaggle/Kaggle_Daily_USD_Market_Data_109_CCs_until_14_05_2025/"
opening_price_df = pd.DataFrame(columns=['TIMESTAMP'])
high_price_df = pd.DataFrame(columns=['TIMESTAMP'])
low_price_df = pd.DataFrame(columns=['TIMESTAMP'])
closing_price_df = pd.DataFrame(columns=['TIMESTAMP'])
launch_date_df = pd.read_csv("All_Crypto_Data/Crypto_Launch_Data/Crypto_Launch_Dates.csv")
launch_date_df['LAUNCH_DATE'] = pd.to_datetime(launch_date_df['LAUNCH_DATE'])

for cc in [f for f in os.listdir(base_data_path) if not f.endswith(".txt")]:
    if cc in list(launch_date_df['CC']):
        if (launch_date_df[launch_date_df['CC'] == cc]['LAUNCH_DATE'].iloc[0] <= datetime(2017, 6, 1)):
            file_name = os.listdir(f"{base_data_path}/{cc}/")[0]
            cc_df = pd.read_csv(f"{base_data_path}/{cc}/{file_name}", 
                                parse_dates=['TIMESTAMP'], 
                                index_col='TIMESTAMP')
            cc_df = cc_df.pct_change().reset_index().iloc[1:]

            opening_price_df = pd.merge(opening_price_df, cc_df[['TIMESTAMP', 
                                                                 f'{cc}_OPEN_PRICE_USD']].rename(columns={f'{cc}_OPEN_PRICE_USD':cc}), 
                                                                 how='outer',
                                                                 on='TIMESTAMP')
            high_price_df = pd.merge(high_price_df, cc_df[['TIMESTAMP', 
                                                         f'{cc}_HIGH_PRICE_USD']].rename(columns={f'{cc}_HIGH_PRICE_USD':cc}), 
                                                         how='outer',
                                                         on='TIMESTAMP')
            low_price_df = pd.merge(low_price_df, cc_df[['TIMESTAMP', 
                                                         f'{cc}_LOW_PRICE_USD']].rename(columns={f'{cc}_LOW_PRICE_USD':cc}), 
                                                         how='outer',
                                                         on='TIMESTAMP')
            closing_price_df = pd.merge(closing_price_df, cc_df[['TIMESTAMP', 
                                                                 f'{cc}_CLOSE_PRICE_USD']].rename(columns={f'{cc}_CLOSE_PRICE_USD':cc}), 
                                                                 how='outer',
                                                                 on='TIMESTAMP')


price_types = {'Opening_Price': opening_price_df[(opening_price_df['TIMESTAMP'] >= datetime(2017, 6, 1)) & (opening_price_df['TIMESTAMP'] < datetime(2025, 1, 1))].drop('TIMESTAMP',
                                                                                                                                                                        axis=1),
               'High_Price': high_price_df[(high_price_df['TIMESTAMP'] >= datetime(2017, 6, 1)) & (high_price_df['TIMESTAMP'] < datetime(2025, 1, 1))].drop('TIMESTAMP',
                                                                                                                                                          axis=1),
               'Low_Price': low_price_df[(low_price_df['TIMESTAMP'] >= datetime(2017, 6, 1)) & (low_price_df['TIMESTAMP'] < datetime(2025, 1, 1))].drop('TIMESTAMP', 
                                                                                                                                                        axis=1),
               'Closing_Price': closing_price_df[(closing_price_df['TIMESTAMP'] >= datetime(2017, 6, 1)) & (closing_price_df['TIMESTAMP'] < datetime(2025, 1, 1))].drop('TIMESTAMP', 
                                                                                                                                                                        axis=1),}
base_save_path_pc = "Exploratory_Data_Analysis/Results/Correlation_Between_Cryptos/109CCs/Retrieved_Correlation/Pearson_Correlation/"
base_save_path_sc = "Exploratory_Data_Analysis/Results/Correlation_Between_Cryptos/109CCs/Retrieved_Correlation/Spearman_Correlation/"
base_save_path_gct = "Exploratory_Data_Analysis/Results/Correlation_Between_Cryptos/109CCs/Retrieved_Correlation/Granger_Causality/"
all_pcs = []
all_scs = []
all_gcts = {'BTC': [],
            'ETH': [],
            'LTC': [],
            'XMR': [],
            'XRP': [],}

if not os.path.exists(base_save_path_pc):
    os.makedirs(base_save_path_pc)

if not os.path.exists(base_save_path_sc):
    os.makedirs(base_save_path_sc)

for pt in price_types.keys():
    current_pt_df = price_types[pt]
    current_pc = current_pt_df.corr(method='pearson')
    current_sc = current_pt_df.corr(method='spearman')

    for cc in all_gcts.keys():
        current_gct = pd.DataFrame()

        for other_cc in [occ for occ in current_pt_df.columns if occ != cc]:
            print(f"{cc}:{other_cc}")

            current_pair_df = current_pt_df[[cc, other_cc]].dropna().reset_index(drop=True)
            
            if (len(current_pair_df) != 0) and (not np.isinf(current_pair_df.to_numpy()).any()):
                current_gct_result = grangercausalitytests(current_pair_df, 
                                                           maxlag=30, 
                                                           verbose=False)
                current_p_values = [lag_res[0]['ssr_chi2test'][1] for lag_res in current_gct_result.values()]
                current_gct[other_cc] = current_p_values
        
        current_gct.index = range(1, 31)
        current_gct.index.name = "Lag"

        all_gcts[cc].append(current_gct)
        current_save_path_gct = f"{base_save_path_gct}{cc}/"

        if not os.path.exists(current_save_path_gct):
            os.makedirs(current_save_path_gct)

        current_gct.to_csv(f'{current_save_path_gct}{cc}_{pt}.csv')

    all_pcs.append(current_pc)
    all_scs.append(current_sc)
    
    current_pc.to_csv(f'{base_save_path_pc}{pt}.csv')
    current_sc.to_csv(f'{base_save_path_sc}{pt}.csv')

pd.concat(all_pcs).groupby(level=0).mean().to_csv(f'{base_save_path_pc}Average.csv')
pd.concat(all_scs).groupby(level=0).mean().to_csv(f'{base_save_path_sc}Average.csv')

for cc in all_gcts.keys():
    current_save_path_gct = f"{base_save_path_gct}{cc}/"
    pd.concat(all_gcts[cc]).groupby(level=0).mean().to_csv(f'{current_save_path_gct}{cc}_Average.csv')

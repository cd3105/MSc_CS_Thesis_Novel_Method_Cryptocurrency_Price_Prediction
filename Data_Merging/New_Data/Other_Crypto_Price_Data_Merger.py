import pandas as pd
import os
from datetime import datetime, timedelta

selected_ccs_base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Binance_Extended/"

other_ccs_investing_base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Other_Investing/"

base_paths = [selected_ccs_base_path, other_ccs_investing_base_path]
ccs_viable_data_available = []

for bp in base_paths:
    print(f"Current Base Path: {bp}")
    print(f"\t- Viable CCs:")

    for cc in [m for m in os.listdir(bp) if not m.endswith('.txt')]:
        current_file = os.listdir(f"{bp}{cc}/1_Day/")[0]
        current_df = pd.read_csv(f"{bp}{cc}/1_Day/{current_file}")
        current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP']).dt.tz_localize(None)

        if (current_df['TIMESTAMP'].min() <= datetime(2017, 6, 1)):
            print(f"\t\t- {cc}")
            ccs_viable_data_available.append(cc)

ccs_viable_data_available = sorted(list(set(ccs_viable_data_available)))
viable_other_ccs_per_selected_cc = {'BTC':[], 
                                    'ETH':[], 
                                    'LTC':[], 
                                    'XMR':[], 
                                    'XRP':[]}
correlation_results_base_path = "Exploratory_Data_Analysis/Results/Correlation_Between_Cryptos/"
significant_p_value_threshold = 0.075

for cc in viable_other_ccs_per_selected_cc.keys():
    for data_type in [m for m in os.listdir(f"{correlation_results_base_path}") if (m == '109CCs') or (m == 'YF_CCs') or (m == 'Binance_via_CoinDesk_CCs')]: # Only Data Covering a Large Period of Time until now
        if cc in os.listdir(f"{correlation_results_base_path}{data_type}/Ranked_Correlation/Granger_Causality/"):
            current_granger_causality_results = pd.read_csv(f"{correlation_results_base_path}{data_type}/Ranked_Correlation/Granger_Causality/{cc}/Closing_Price/{cc}_Ranked_by_Mean_Closing_Price_Granger_Causality.csv",
                                                            index_col=0)

            current_significant_granger_causality_results = current_granger_causality_results[current_granger_causality_results[cc] <= significant_p_value_threshold]
            current_significant_granger_causality_results = current_significant_granger_causality_results[current_significant_granger_causality_results.index.isin(ccs_viable_data_available)]
           
            current_list = viable_other_ccs_per_selected_cc[cc]
           
            for row in current_significant_granger_causality_results.iterrows():
                current_list.append((row[0], row[1][cc]))
                viable_other_ccs_per_selected_cc[cc] = current_list


for cc in viable_other_ccs_per_selected_cc.keys():
    current_list = viable_other_ccs_per_selected_cc[cc]

    unique_ccs = list(set([i[0] for i in current_list]))
    new_list = []

    for ucc in unique_ccs:
        if ucc != cc:
            new_list.append((ucc, min([i[1] for i in current_list if i[0] == ucc])))

    viable_other_ccs_per_selected_cc[cc] = sorted(new_list, key=lambda x: x[1])


selected_start_date = datetime(2017, 6, 1)
selected_end_date = datetime(2025, 1, 1)
selected_date_range = pd.date_range(selected_start_date,
                                    selected_end_date - timedelta(days=1),
                                    freq='D')
selected_date_range_df = pd.DataFrame({'TIMESTAMP':selected_date_range})
frequencies = ['1_Hour', '2_Hour', '4_Hour', '6_Hour', '8_Hour', '12_Hour', '1_Day']

for cc in viable_other_ccs_per_selected_cc.keys():
    for f in frequencies:
        current_occ_price_data_base_path = f"All_Crypto_Data/New_Data/Processed/Other_Crypto_Price_Data/{cc}/{f}/"
        current_all_occ_df = pd.DataFrame({'TIMESTAMP':[]})
        ranking = 1

        if not os.path.exists(current_occ_price_data_base_path):
            os.makedirs(current_occ_price_data_base_path)

        for occ, p_val in viable_other_ccs_per_selected_cc[cc]:
            if f == '1_Day':
                if occ in viable_other_ccs_per_selected_cc.keys():
                    current_occ_path = f"{selected_ccs_base_path}{occ}/1_Day/"
                    current_occ_fn = os.listdir(current_occ_path)[0]
                    current_occ_df = pd.read_csv(f"{current_occ_path}{current_occ_fn}")[["TIMESTAMP", 
                                                                                        f"{occ}_OPEN_PRICE_USD",  
                                                                                        f"{occ}_HIGH_PRICE_USD", 
                                                                                        f"{occ}_LOW_PRICE_USD",
                                                                                        f"{occ}_CLOSE_PRICE_USD",]]
                    current_occ_df['TIMESTAMP'] = pd.to_datetime(current_occ_df['TIMESTAMP'])
                    current_occ_df = pd.merge(selected_date_range_df,
                                            current_occ_df,
                                            on='TIMESTAMP',
                                            how='left')
                else:
                    if occ in os.listdir("All_Crypto_Data/Crypto_Market_Data/Merged/Other_Binance_via_CoinDesk/"):
                        current_binance_occ_path = f"All_Crypto_Data/Crypto_Market_Data/Merged/Other_Binance_via_CoinDesk/{occ}/1_Day/"
                        current_binance_occ_fn = os.listdir(current_binance_occ_path)[0]
                        current_binance_occ_df = pd.read_csv(f"{current_binance_occ_path}{current_binance_occ_fn}")[["TIMESTAMP", 
                                                                                                                    f"{occ}_OPEN_PRICE_USDT", 
                                                                                                                    f"{occ}_HIGH_PRICE_USDT", 
                                                                                                                    f"{occ}_LOW_PRICE_USDT",
                                                                                                                    f"{occ}_CLOSE_PRICE_USDT",]].rename(columns={f"{occ}_OPEN_PRICE_USDT":f"{occ}_OPEN_PRICE_USD", 
                                                                                                                                                                f"{occ}_HIGH_PRICE_USDT":f"{occ}_HIGH_PRICE_USD", 
                                                                                                                                                                f"{occ}_LOW_PRICE_USDT":f"{occ}_LOW_PRICE_USD",
                                                                                                                                                                f"{occ}_CLOSE_PRICE_USDT":f"{occ}_CLOSE_PRICE_USD",})
                        current_binance_occ_df['TIMESTAMP'] = pd.to_datetime(current_binance_occ_df['TIMESTAMP'])
                        current_binance_occ_df = pd.merge(selected_date_range_df,
                                                        current_binance_occ_df,
                                                        on='TIMESTAMP',
                                                        how='left')
                        
                        missing_dates = current_binance_occ_df[current_binance_occ_df.isna().any(axis=1)]['TIMESTAMP']
                        
                        current_investing_occ_path = f"All_Crypto_Data/Crypto_Market_Data/Merged/Other_Investing/{occ}/1_Day/"
                        current_investing_occ_fn = os.listdir(current_investing_occ_path)[0]
                        current_investing_occ_df = pd.read_csv(f"{current_investing_occ_path}{current_investing_occ_fn}")[["TIMESTAMP",
                                                                                                                        f"{occ}_OPEN_PRICE_USD", 
                                                                                                                        f"{occ}_HIGH_PRICE_USD", 
                                                                                                                        f"{occ}_LOW_PRICE_USD",
                                                                                                                        f"{occ}_CLOSE_PRICE_USD",]]
                        current_investing_occ_df['TIMESTAMP'] = pd.to_datetime(current_investing_occ_df['TIMESTAMP']) 
                        
                        current_occ_df = pd.concat([current_investing_occ_df[current_investing_occ_df['TIMESTAMP'].isin(missing_dates)], 
                                                    current_binance_occ_df[~current_binance_occ_df['TIMESTAMP'].isin(missing_dates)]])
                    else:
                        current_investing_occ_path = f"All_Crypto_Data/Crypto_Market_Data/Merged/Other_Investing/{occ}/1_Day/"
                        current_investing_occ_fn = os.listdir(current_investing_occ_path)[0]
                        current_investing_occ_df = pd.read_csv(f"{current_investing_occ_path}{current_investing_occ_fn}")[["TIMESTAMP",
                                                                                                                        f"{occ}_OPEN_PRICE_USD", 
                                                                                                                        f"{occ}_HIGH_PRICE_USD", 
                                                                                                                        f"{occ}_LOW_PRICE_USD",
                                                                                                                        f"{occ}_CLOSE_PRICE_USD",]]
                        current_investing_occ_df['TIMESTAMP'] = pd.to_datetime(current_investing_occ_df['TIMESTAMP']) 
                        current_investing_occ_df = pd.merge(selected_date_range_df,
                                                            current_investing_occ_df,
                                                            on='TIMESTAMP',
                                                            how='left')

                        missing_dates = current_investing_occ_df[current_investing_occ_df.isna().any(axis=1)]['TIMESTAMP']

                        current_yf_occ_path = f"All_Crypto_Data/Crypto_Market_Data/Merged/Yahoo_Finance/{occ}/1_Day/"
                        current_yf_occ_fn = os.listdir(current_yf_occ_path)[0]
                        current_yf_occ_df = pd.read_csv(f"{current_yf_occ_path}{current_yf_occ_fn}")[["TIMESTAMP",
                                                                                                    f"{occ}_OPEN_PRICE_USD", 
                                                                                                    f"{occ}_HIGH_PRICE_USD", 
                                                                                                    f"{occ}_LOW_PRICE_USD",
                                                                                                    f"{occ}_CLOSE_PRICE_USD",]]
                        current_yf_occ_df['TIMESTAMP'] = pd.to_datetime(current_yf_occ_df['TIMESTAMP']) 

                        current_occ_df = pd.concat([current_yf_occ_df[current_yf_occ_df['TIMESTAMP'].isin(missing_dates)], 
                                                    current_investing_occ_df[~current_investing_occ_df['TIMESTAMP'].isin(missing_dates)]])    
            else:
                if occ in viable_other_ccs_per_selected_cc.keys():
                    current_occ_path = f"All_Crypto_Data/Crypto_Market_Data/Merged/Binance_via_CoinDesk/{occ}/{f}/"
                else:
                    current_occ_path = f"All_Crypto_Data/Crypto_Market_Data/Merged/Other_Binance_via_CoinDesk/{occ}/{f}/"
                    
                current_occ_fn = os.listdir(current_occ_path)[0]
                current_occ_df = pd.read_csv(f"{current_occ_path}{current_occ_fn}")[["TIMESTAMP",
                                                                                     f"{occ}_OPEN_PRICE_USDT",  
                                                                                     f"{occ}_HIGH_PRICE_USDT", 
                                                                                     f"{occ}_LOW_PRICE_USDT",
                                                                                     f"{occ}_CLOSE_PRICE_USDT",]]
                current_occ_df['TIMESTAMP'] = pd.to_datetime(current_occ_df['TIMESTAMP'])

            current_occ_df['TIMESTAMP'] = pd.to_datetime(current_occ_df['TIMESTAMP']) 
            current_all_occ_df = pd.merge(current_all_occ_df,
                                          current_occ_df,
                                          on="TIMESTAMP",
                                          how="outer")
            
            if f == '1_Day':
                current_occ_df.to_csv(f"{current_occ_price_data_base_path}{ranking}_{occ}_USD_Daily_Price_Data_{current_occ_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_occ_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                      index=False)
            elif f == '1_Hour':
                current_occ_df.to_csv(f"{current_occ_price_data_base_path}{ranking}_{occ}_USD_Hourly_Price_Data_{current_occ_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_occ_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                      index=False)
            else:
                current_occ_df.to_csv(f"{current_occ_price_data_base_path}{ranking}_{occ}_USD_{f}ly_Price_Data_{current_occ_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_occ_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                      index=False)
                
            ranking+=1
        
        if f == '1_Day':
            current_all_occ_df.to_csv(f"{current_occ_price_data_base_path}All_Other_Crypto_USD_Daily_Price_Data_{current_all_occ_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_all_occ_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                    index=False)
        elif f == '1_Hour':
            current_all_occ_df.to_csv(f"{current_occ_price_data_base_path}All_Other_Crypto_USD_Hourly_Price_Data_{current_all_occ_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_all_occ_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                    index=False)
        else:
            current_all_occ_df.to_csv(f"{current_occ_price_data_base_path}All_Other_Crypto_USD_{f}ly_Price_Data_{current_all_occ_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{current_all_occ_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv", 
                                    index=False)    


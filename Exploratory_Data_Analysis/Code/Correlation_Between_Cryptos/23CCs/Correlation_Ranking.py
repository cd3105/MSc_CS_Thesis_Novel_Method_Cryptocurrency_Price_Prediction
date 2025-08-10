import pandas as pd
import os

correlation_base_path = "Exploratory_Data_Analysis/Results/Correlation_Between_Cryptos/23CCs/Retrieved_Correlation/"
selected_cryptos = ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']

for corr in os.listdir(correlation_base_path):
    print(f"Correlation Type: {corr}")

    if corr == 'Granger_Causality':
        for cc in selected_cryptos:
            print(f"\t- Crypto: {cc}")

            for csv in os.listdir(f"{correlation_base_path}{corr}/{cc}/"):
                print(f"\t\t- File: {csv[4:-4]}")

                current_corr_df = pd.read_csv(f"{correlation_base_path}{corr}/{cc}/{csv}", 
                                              index_col=0)
                current_min_ranked_corr_df = pd.DataFrame()
                current_mean_ranked_corr_df = pd.DataFrame()
                current_median_ranked_corr_df = pd.DataFrame()
                
                current_min_ranked_corr_df.index = list(current_corr_df.columns)
                current_mean_ranked_corr_df.index = list(current_corr_df.columns)
                current_median_ranked_corr_df.index = list(current_corr_df.columns)

                current_min_ranked_corr_df[cc] = [current_corr_df[c].min() for c in current_corr_df.columns]
                current_mean_ranked_corr_df[cc] = [current_corr_df[c].mean() for c in current_corr_df.columns]
                current_median_ranked_corr_df[cc] = [current_corr_df[c].median() for c in current_corr_df.columns]

                current_min_ranked_corr_df['Lag'] = [current_corr_df[c].idxmin() for c in current_corr_df.columns]
                
                current_ranked_corr_fp = f"Exploratory_Data_Analysis/Results/Correlation_Between_Cryptos/23CCs/Ranked_Correlation/{corr}/{cc}/{csv[4:-4]}/"

                if not os.path.exists(current_ranked_corr_fp):
                    os.makedirs(current_ranked_corr_fp)
                
                current_mean_ranked_corr_df.sort_values(cc, 
                                                        ascending=True).to_csv(f"{current_ranked_corr_fp}{cc}_Ranked_by_Mean_{csv[4:-4]}_{corr}.csv")
                current_median_ranked_corr_df.sort_values(cc, 
                                                          ascending=True).to_csv(f"{current_ranked_corr_fp}{cc}_Ranked_by_Median_{csv[4:-4]}_{corr}.csv")
                current_min_ranked_corr_df.sort_values(cc, 
                                                       ascending=True).to_csv(f"{current_ranked_corr_fp}{cc}_Ranked_by_Min_{csv[4:-4]}_{corr}.csv")
    else:
        for csv in os.listdir(f"{correlation_base_path}{corr}/"):
            print(f"\t- File: {csv[:-4]}")

            for cc in selected_cryptos:
                print(f"\t\t- Crypto: {cc}")
                current_ranked_corr_df = pd.read_csv(f"{correlation_base_path}{corr}/{csv}", 
                                                    index_col=0)[[cc]].sort_values(cc, 
                                                                                    ascending=False).iloc[1:]
                current_ranked_corr_fp = f"Exploratory_Data_Analysis/Results/Correlation_Between_Cryptos/23CCs/Ranked_Correlation/{corr}/{cc}/"

                if not os.path.exists(current_ranked_corr_fp):
                    os.makedirs(current_ranked_corr_fp)

                current_ranked_corr_df.to_csv(f"{current_ranked_corr_fp}{cc}_Ranked_{csv[:-4]}_{corr}.csv")

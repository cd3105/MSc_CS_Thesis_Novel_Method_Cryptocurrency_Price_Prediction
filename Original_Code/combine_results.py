import pandas as pd
import numpy as np
import os

paths = ['Original_Code/old_res_rerun/', 'Original_Code/new_res/']
cryptocurrencies = ['btc', 'eth', 'ltc', 'xmr', 'xrp']

index_mapping = {0:'June 2021',
                 1:'July 2021',
                 2:'August 2021',
                 3:'September 2021',
                 4:'October 2021',
                 5:'November 2021',
                 6:'December 2021',
                 7:'January 2022',
                 8:'February 2022',
                 9:'March 2022',
                 10:'April 2022',
                 11:'May 2022',
                 12:'Mean',}

for c in cryptocurrencies:
    for p in paths:
        test_periods = []
        models = []
        normalized_rmses = []
        normalized_maes = []
        normalized_mapes = []
        rmses = []
        maes = []
        mapes = []
        r2s = []
        
        for i in list(index_mapping.keys()):
            all_models = [m for m in os.listdir(p) if m != 'COMBINED']

            for m_i in range(len(all_models)):
                current_results_base_path = f"{p}{all_models[m_i]}/Average_Metrics/"
                current_normalized_results = pd.read_csv(f"{p}{all_models[m_i]}/Average_Metrics/Normalized/average_metrics_{all_models[m_i].lower()}-{c}-close-w30-h0-12m_N.csv", 
                                                         index_col=0)
                current_regular_results = pd.read_csv(f"{p}{all_models[m_i]}/Average_Metrics/Regular/average_metrics_{all_models[m_i].lower()}-{c}-close-w30-h0-12m.csv", 
                                                      index_col=0)

                if m_i == 0:
                    test_periods.append(index_mapping[i])
                else:
                    test_periods.append('')
                
                models.append(all_models[m_i])
                normalized_rmses.append(f"{current_normalized_results.iloc[i]['RMSE']:.4f}")
                normalized_maes.append(f"{current_normalized_results.iloc[i]['MAE']:.4f}")
                normalized_mapes.append(f"{current_normalized_results.iloc[i]['MAPE'] * 100:.3f}")

                if c == 'xrp':
                    rmses.append(f"{current_regular_results.iloc[i]['RMSE']:.4f}")
                    maes.append(f"{current_regular_results.iloc[i]['MAE']:.4f}")
                else:
                    rmses.append(f"{current_regular_results.iloc[i]['RMSE']:.3f}")
                    maes.append(f"{current_regular_results.iloc[i]['MAE']:.3f}")

                mapes.append(f"{current_regular_results.iloc[i]['MAPE'] * 100:.3f}")
                r2s.append(f"{current_normalized_results.iloc[i]['R2']:.3f}")
        
        current_combined_res_df = pd.DataFrame({'Test Period':test_periods,
                                                'Model':models,
                                                'nRMSE':normalized_rmses,
                                                'nMAE':normalized_maes,
                                                'nMAPE':normalized_mapes,
                                                'RMSE':rmses,
                                                'MAE':maes,
                                                'MAPE':mapes,
                                                'R2':r2s})
        current_combined_res_path = f"{p}COMBINED/"

        if not os.path.exists(current_combined_res_path):
            os.makedirs(current_combined_res_path)

        current_combined_res_df.to_csv(f"{current_combined_res_path}combined_metrics_{c}.csv", 
                                       index=False)
        current_combined_res_df.to_latex(f"{current_combined_res_path}combined_metrics_{c}.tex", 
                                         index=False)

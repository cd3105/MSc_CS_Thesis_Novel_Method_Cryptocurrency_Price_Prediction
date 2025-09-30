from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
from datetime import datetime
import os
import warnings

warnings.filterwarnings("ignore")

experiments = {'Baseline':'All_Crypto_Data/Adopted_Data/Processed/Baseline/',
               'Extended_Binance':'All_Crypto_Data/Adopted_Data/Processed/Extended/',
               'Full_Binance':'All_Crypto_Data/Adopted_Data/Processed/Full/',}

def non_stationarity_determination_adf(ts,
                                       p_val,
                                       cv_5):
    ts_cv_determination = True
    p_val_determination = True
    significance_level = 0.05

    if ts < cv_5:
        ts_cv_determination = False
    
    if p_val <= significance_level:
        p_val_determination = False

    if ts_cv_determination and p_val_determination:
        return "Non-Stationary"
    else:
        return "Stationary"
    

def non_stationarity_determination_kpss(ts,
                                        p_val,
                                        cv_5):
    
    ts_cv_determination = False
    p_val_determination = False
    significance_level = 0.05

    if ts > cv_5:
        ts_cv_determination = True
    
    if p_val < significance_level:
        p_val_determination = True

    if ts_cv_determination and p_val_determination:
        return "Non-Stationary"
    else:
        return "Stationary"


for e in experiments.keys():
    current_price_data_base_path = f'{experiments[e]}Price_Data/Binance/'

    for cc in os.listdir(current_price_data_base_path):
        print(f"Current Crypto: {cc}")

        price_column_mapping = {f'{cc}_OPEN_PRICE_USDT':'Opening_Price',
                                f'{cc}_HIGH_PRICE_USDT':'High_Price',
                                f'{cc}_LOW_PRICE_USDT':'Low_Price',
                                f'{cc}_CLOSE_PRICE_USDT':'Closing_Price',}
        for pc in price_column_mapping.keys():
            print(f"\t- {e}:")
            print(f"\t\t- {pc}:")

            current_frequencies = []
            current_adf_test_statistics = []
            current_adf_p_vals = []
            current_adf_critical_vals_1 = []
            current_adf_critical_vals_5 = []
            current_adf_critical_vals_10 = []
            current_adf_determinations = []
            
            current_kpss_test_statistics = []
            current_kpss_p_vals = []
            current_kpss_critical_vals_1 = []
            current_kpss_critical_vals_2_5 = []
            current_kpss_critical_vals_5 = []
            current_kpss_critical_vals_10 = []
            current_kpss_determinations = []

            for f in os.listdir(f'{current_price_data_base_path}{cc}'):
                print(f"\t\t\t- Current Frequency: {f}")

                current_file = os.listdir(f"{current_price_data_base_path}{cc}/{f}/")[0]
                current_df = pd.read_csv(f"{current_price_data_base_path}{cc}/{f}/{current_file}")
                current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])
                    
                current_adf_result = adfuller(current_df[pc])
                current_kpss_result = kpss(current_df[pc])

                current_adf_determination = non_stationarity_determination_adf(ts=current_adf_result[0],
                                                                               p_val=current_adf_result[1],
                                                                               cv_5=current_adf_result[4]['5%'])
                current_kpss_determination = non_stationarity_determination_kpss(ts=current_kpss_result[0],
                                                                                 p_val=current_kpss_result[1],
                                                                                 cv_5=current_kpss_result[3]['5%'])

                current_frequencies.append(f)
                current_adf_test_statistics.append(f"{current_adf_result[0]:.3f}")
                current_adf_p_vals.append(f"{current_adf_result[1]:.3f}")
                current_adf_critical_vals_1.append(f"{current_adf_result[4]['1%']:.3f}")
                current_adf_critical_vals_5.append(f"{current_adf_result[4]['5%']:.3f}")
                current_adf_critical_vals_10.append(f"{current_adf_result[4]['10%']:.3f}")
                current_adf_determinations.append(current_adf_determination)

                current_kpss_test_statistics.append(f"{current_kpss_result[0]:.3f}")
                current_kpss_p_vals.append(f"{current_kpss_result[1]:.3f}")
                current_kpss_critical_vals_1.append(f"{current_kpss_result[3]['1%']:.3f}")
                current_kpss_critical_vals_2_5.append(f"{current_kpss_result[3]['2.5%']:.3f}")
                current_kpss_critical_vals_5.append(f"{current_kpss_result[3]['5%']:.3f}")
                current_kpss_critical_vals_10.append(f"{current_kpss_result[3]['10%']:.3f}")
                current_kpss_determinations.append(current_kpss_determination)

                print(f"\t\t\t\t - ADF P-Val Results: {current_adf_result[1]}")
                print(f"\t\t\t\t - Determination based on ADF: {current_adf_determination}")
                print(f"\t\t\t\t - KPSS P-Val Results: {current_kpss_result[1]}")
                print(f"\t\t\t\t - Determination based on KPSS: {current_kpss_determination}")
                
            current_results_df = pd.DataFrame({'Frequency':current_frequencies,
                                               'ADF TS':current_adf_test_statistics,
                                               'ADF P-Val.':current_adf_p_vals,
                                               'ADF CV (1\%)':current_adf_critical_vals_1,
                                               'ADF CV (5\%)':current_adf_critical_vals_5,
                                               'ADF CV (10\%)':current_adf_critical_vals_10,
                                               'ADF_Determination':current_adf_determinations,

                                               'KPSS TS':current_kpss_test_statistics,
                                               'KPSS P-Val':current_kpss_p_vals,
                                               'KPSS CV (1\%)':current_kpss_critical_vals_1,
                                               'KPSS CV (2.5\%)':current_kpss_critical_vals_2_5,
                                               'KPSS CV (5\%)':current_kpss_critical_vals_5,
                                               'KPSS CV (10\%)':current_kpss_critical_vals_10,
                                               'KPSS_Determination':current_kpss_determinations,})
                
            current_results_path = f"Exploratory_Data_Analysis/Results/Stationarity_Check/{e}/{cc}/"

            if not os.path.exists(current_results_path):
                os.makedirs(current_results_path)

            current_results_df.to_csv(f"{current_results_path}Stationary_Check_{cc}_{price_column_mapping[pc]}.csv")
            current_results_df.to_latex(f"{current_results_path}Stationary_Check_{cc}_{price_column_mapping[pc]}.tex")


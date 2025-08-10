from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
from datetime import datetime
import os
import warnings

warnings.filterwarnings("ignore")
market_data_base_path = "All_Crypto_Data/New_Data/Processed/Market_Data/All_Market_Data/"

for cc in os.listdir(market_data_base_path):
    print(f"Current Crypto: {cc}")

    price_column_mapping = {f'{cc}_OPEN_PRICE_USD':'Opening_Price',
                            f'{cc}_HIGH_PRICE_USD':'High_Price',
                            f'{cc}_LOW_PRICE_USD':'Low_Price',
                            f'{cc}_CLOSE_PRICE_USD':'Closing_Price',}
    for pc in price_column_mapping.keys():
        print(f"\t- {pc}:")

        periods = ['Initial_Period',
                   'Extended_Period']

        for p in periods:
            print(f"\t\t- {p}:")

            current_frequencies = []
            current_adf_p_vals = []
            current_adf_p_determinations = []
            current_kpss_p_vals = []
            current_kpss_p_determinations = []

            for f in os.listdir(f"{market_data_base_path}{cc}"):
                print(f"\t\t\t- Current Frequency: {f}")

                current_file = os.listdir(f"{market_data_base_path}{cc}/{f}/")[0]
                current_df = pd.read_csv(f"{market_data_base_path}{cc}/{f}/{current_file}")
                current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])

                if p == 'Initial_Period':
                    current_df = current_df[current_df['TIMESTAMP'] <= datetime(2022, 5, 31, 23)]
                
                current_adf_result = adfuller(current_df[pc])
                current_kpss_result = kpss(current_df[pc])

                current_frequencies.append(f)
                current_adf_p_vals.append(current_adf_result[1])
                current_adf_p_determinations.append('Non-Stationary' if current_adf_result[1] >= 0.05 else 'Stationary')
                current_kpss_p_vals.append(current_kpss_result[1])
                current_kpss_p_determinations.append('Stationary' if current_kpss_result[1] >= 0.05 else 'Non-Stationary')

                print(f"\t\t\t\t - ADF P-Val Results: {current_adf_result[1]}")
                print(f"\t\t\t\t - Determination based on ADF P-Val: {'Non-Stationary' if current_adf_result[1] >= 0.05 else 'Stationary'}")
                print(f"\t\t\t\t - KPSS P-Val Results: {current_kpss_result[1]}")
                print(f"\t\t\t\t - Determination based on KPSS P-Val: {'Stationary' if current_kpss_result[1] >= 0.05 else 'Non-Stationary'}")
            
            current_results_df = pd.DataFrame({'Frequency':current_frequencies,
                                               'ADF_P_Val':current_adf_p_vals,
                                               'ADF_Determination':current_adf_p_determinations,
                                               'KPSS_P_Val':current_kpss_p_vals,
                                               'KPSS_Determination':current_kpss_p_determinations,})
            
            current_results_path = f"Exploratory_Data_Analysis/Results/Stationarity_Check/{cc}/{p}/"

            if not os.path.exists(current_results_path):
                os.makedirs(current_results_path)

            current_results_df.to_csv(f"{current_results_path}Stationary_Check_{p}_{cc}_{price_column_mapping[pc]}.csv")


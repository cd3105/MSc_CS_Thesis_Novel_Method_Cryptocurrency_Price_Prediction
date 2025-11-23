import os
import pandas as pd
import numpy as np
from pysdkit import VMD
from itertools import product
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import kpss

Ks = list(range(2, 16))
alpha_vals = list(range(250, 8001, 250))
tau_vals =  [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

experiments = {
    'Baseline':'All_Crypto_Data/Adopted_Data/Processed/Baseline/',
    'Extended_Binance':'All_Crypto_Data/Adopted_Data/Processed/Extended/',
    'Full_Binance':'All_Crypto_Data/Adopted_Data/Processed/Full/',
}

combinations = list(
    product(
        Ks, 
        alpha_vals, 
        tau_vals,
    )
)

print(f'Total Combinations: {len(combinations)}\n')

for e in experiments.keys():
    print(f'Experiment: {e}')

    current_price_data_base_path = f'{experiments[e]}Price_Data/Binance/'

    for cc in os.listdir(current_price_data_base_path):
        print(f'- Crypto: {cc}')

        for f in os.listdir(f"{current_price_data_base_path}{cc}"):
            print(f'\t- Frequency: {f}')

            if f == '1_Day': 
                current_file = os.listdir(f"{current_price_data_base_path}{cc}/{f}/")[0]
                current_df = pd.read_csv(f"{current_price_data_base_path}{cc}/{f}/{current_file}")
                current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])
                imf_1_p_values = []
                non_stationarity_imfs_counters = []
                mses = []
                significance_level = 0.05

                for i, (k, alpha, tau) in enumerate(combinations):
                    if e == 'Full_Binance':
                        original_close_price_ts = np.array(current_df[f'{cc}_CLOSE_PRICE_USDT'])[:-1]
                    else:
                        original_close_price_ts = np.array(current_df[f'{cc}_CLOSE_PRICE_USDT'])

                    vmd = VMD(K=k,
                              alpha=alpha,
                              tau=tau)
                    imfs = vmd.fit_transform(signal=original_close_price_ts)

                    reconstructed_close_price_ts = imfs.sum(axis=0)
                    non_stationarity_imfs_counter = 0

                    for j, imf in enumerate(imfs):
                        current_imf_p_val = kpss(imf)[1]

                        if current_imf_p_val <= significance_level:
                            non_stationarity_imfs_counter+=1

                        if j == 0:
                            imf_1_p_values.append(current_imf_p_val)
                    
                    non_stationarity_imfs_counters.append(non_stationarity_imfs_counter)
                    mses.append(
                        mean_squared_error(
                            y_true=original_close_price_ts, 
                            y_pred=reconstructed_close_price_ts,
                        )
                    )

                    if ((i+1) % 250) == 0:
                        print(f'\t- Reviewed Combinations: {i+1}')

                current_results_df = pd.DataFrame(
                    {
                        'K':[t[0] for t in combinations],
                        'Alpha':[t[1] for t in combinations],
                        'Tau':[t[2] for t in combinations],
                        'Non_Stationary_IMFs':non_stationarity_imfs_counters,
                        'MSE':mses,
                        'IMF_1_P_Val':imf_1_p_values,
                    }
                ).sort_values(by=['Non_Stationary_IMFs', 'MSE', 'IMF_1_P_Val'], ascending=[True, True, False]).reset_index(drop=True)

                current_results_path = f"Exploratory_Data_Analysis/Results/Optimal_VMD/Stationarity_Based/Full/{e}/{cc}/"

                if not os.path.exists(current_results_path):
                    os.makedirs(current_results_path)

                if f == '1_Day':
                    current_results_df.to_csv(f"{current_results_path}KPSS_Stationarity_{cc}_Daily_Close_Price.csv")

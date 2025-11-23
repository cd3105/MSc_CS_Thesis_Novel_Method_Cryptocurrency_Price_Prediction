import os
import pandas as pd
import numpy as np
from pysdkit import VMD
from itertools import product
from sklearn.metrics import mean_squared_error


Ks = list(range(2, 16))
alpha_vals = list(range(250, 8001, 250))
tau_vals =  [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
window_sizes = [6, 14, 20, 30, 60, 90, 120, 150, 180, 360]

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
        window_sizes,
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
                mses = []

                for i, (k, alpha, tau, w) in enumerate(combinations):
                    original_close_price_ts = np.array(current_df[f'{cc}_CLOSE_PRICE_USDT'])
                    current_mses = []

                    for j in range(len(original_close_price_ts) - w):
                        current_close_price_ts = original_close_price_ts[j:j+w]

                        vmd = VMD(
                            K=k,
                            alpha=alpha,
                            tau=tau,
                        )
                        
                        imfs = vmd.fit_transform(signal=current_close_price_ts)                    
                        current_reconstructed_close_price_ts = imfs.sum(axis=0)

                        current_mses.append(
                            mean_squared_error(
                                y_true=current_close_price_ts, 
                                y_pred=current_reconstructed_close_price_ts,
                            )
                        )

                    mses.append(np.array(current_mses).mean())

                    if ((i+1) % 250) == 0:
                        print(f'\t- Reviewed Combinations: {i+1}')

                current_results_df = pd.DataFrame(
                    {
                        'K':[t[0] for t in combinations],
                        'Alpha':[t[1] for t in combinations],
                        'Tau':[t[2] for t in combinations],
                        'Window_Size':[t[3] for t in combinations],
                        'MSE':mses
                    }
                ).sort_values(by='MSE').reset_index(drop=True)

                current_results_path = f"Exploratory_Data_Analysis/Results/Optimal_VMD/Reconstruction_Based/Windowed/{e}/{cc}/"

                if not os.path.exists(current_results_path):
                    os.makedirs(current_results_path)

                if f == '1_Day':
                    current_results_df.to_csv(f"{current_results_path}VMD_2_Reconstruction_Accuracy_{cc}_Daily_Close_Price.csv")

import pandas as pd
import numpy as np
from pysdkit import VMD, CEEMDAN
from pysdkit.plot import plot_IMFs_amplitude_spectra, plot_IMFs
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from datetime import datetime
import time

BTC_price_df = pd.read_csv("All_Crypto_Data/Adopted_Data/Processed/Baseline/Price_Data/Binance/BTC/1_Day/BTC_Price_Data.csv")
BTC_price_df['TIMESTAMP'] = pd.to_datetime(BTC_price_df['TIMESTAMP'])
initial_split_date = datetime(2021, 6, 1)
train_test_split_idx = list(BTC_price_df['TIMESTAMP']).index(initial_split_date)

BTC_price_df = BTC_price_df.drop(columns=['TIMESTAMP'])
close_price_ts = np.array(BTC_price_df[f'BTC_CLOSE_PRICE_USDT'])

vmd = VMD(K=13,
          alpha=2750,
          tau=0.05)

ceemdan = CEEMDAN(trials=200,
                  epsilon=2.0,
                  max_imfs=15)

close_price_ts_train = close_price_ts[:train_test_split_idx]
close_price_ts_test = close_price_ts[train_test_split_idx:]

if (len(close_price_ts_train) % 2) != 0:
    close_price_ts_train = close_price_ts_train[1:]

vmd_sm_start = time.time()

decomposed_close_price_IMFs_train = vmd.fit_transform(signal=close_price_ts_train)

for i in range(1, 30):
    #print(f'Single Month: {i+1}')
    current_close_price_ts = np.concatenate([
        close_price_ts_train,#[i:], 
        close_price_ts_test[:i],
    ])[-i-120:-i]
    #print(f'Current Shape: {current_close_price_ts}')
    current_decomposed_close_price_IMFs = vmd.fit_transform(signal=current_close_price_ts)

vmd_sm_end = time.time()

print(f'\nTIME ELAPSED VMD (SINGLE MONTH): {int((vmd_sm_end-vmd_sm_start) // 3600)}h {int(((vmd_sm_end-vmd_sm_start) % 3600) // 60)}m {int((vmd_sm_end-vmd_sm_start) // 60)}s')

#print('\n')

vmd_start = time.time()

decomposed_close_price_IMFs_train = vmd.fit_transform(signal=close_price_ts_train)

for i in range(1, 365):
    #print(f'Full Year: {i+1}')
    current_close_price_ts = np.concatenate([
        close_price_ts_train,#[i:], 
        close_price_ts_test[:i],
    ])[-i-120:-i]
    #print(f'Current Shape: {current_close_price_ts.shape}')
    current_decomposed_close_price_IMFs = vmd.fit_transform(signal=current_close_price_ts)

vmd_end = time.time()

#print('\n')

print(f'\nTIME ELAPSED VMD (ALL MONTHS): {int((vmd_end-vmd_start) // 3600)}h {int(((vmd_end-vmd_start) % 3600) // 60)}m {int((vmd_end-vmd_start) // 60)}s')

#decomposed_close_price_IMFs_train = vmd.fit_transform(signal=close_price_ts_train)
vmd_full_start = time.time()

for i in range(len(close_price_ts_train)+len(close_price_ts_test) - 30):
    #print(f'Full Set: {i+1}')
    current_close_price_ts = np.concatenate([
        close_price_ts_train, 
        close_price_ts_test,
    ])[i:i+120]
    #print(f'Current Shape: {current_close_price_ts.shape}')
    current_decomposed_close_price_IMFs = vmd.fit_transform(signal=current_close_price_ts)

vmd_full_end = time.time()

print(f'\nTIME ELAPSED VMD (FULL WINDOWED SET): {int((vmd_full_end-vmd_full_start) // 3600)}h {int(((vmd_full_end-vmd_full_start) % 3600) // 60)}m {int((vmd_full_end-vmd_full_start) // 60)}s')

# vmd_n_ceemdan_sm_start = time.time()

# decomposed_close_price_IMFs_train = vmd.fit_transform(signal=close_price_ts_train)
# close_price_residual_train = close_price_ts_train.copy()

# for imf in decomposed_close_price_IMFs_train:
#     close_price_residual_train -= imf

# decomposed_close_price_residual_IMFs_train = ceemdan.fit_transform(signal=close_price_residual_train)

# for i in range(1, 30):
#     current_close_price_ts = np.concatenate([
#         close_price_ts_train[i:], 
#         close_price_ts_test[:i],
#     ])
#     current_decomposed_close_price_IMFs = vmd.fit_transform(signal=current_close_price_ts)
#     current_close_price_residual = current_close_price_ts.copy()

#     for imf in current_decomposed_close_price_IMFs:
#         current_close_price_residual -= imf

#     decomposed_close_price_residual_IMFs = ceemdan.fit_transform(signal=current_close_price_residual)

# vmd_n_ceemdan_sm_end = time.time()

# vmd_n_ceemdan_start = time.time()

# decomposed_close_price_IMFs_train = vmd.fit_transform(signal=close_price_ts_train)
# close_price_residual_train = close_price_ts_train.copy()

# for imf in decomposed_close_price_IMFs_train:
#     close_price_residual_train -= imf

# decomposed_close_price_residual_IMFs_train = ceemdan.fit_transform(signal=close_price_residual_train)

# for i in range(1, 365):
#     current_close_price_ts = np.concatenate([
#         close_price_ts_train[i:], 
#         close_price_ts_test[:i],
#     ])
#     current_decomposed_close_price_IMFs = vmd.fit_transform(signal=current_close_price_ts)
#     current_close_price_residual = current_close_price_ts.copy()

#     for imf in current_decomposed_close_price_IMFs:
#         current_close_price_residual -= imf

#     decomposed_close_price_residual_IMFs = ceemdan.fit_transform(signal=current_close_price_residual)

# vmd_n_ceemdan_end = time.time()

# print(f'\nTIME ELAPSED VMD AND CEEMDAN (SINGLE MONTH): {int((vmd_n_ceemdan_sm_end-vmd_n_ceemdan_sm_start) // 3600)}h {int(((vmd_n_ceemdan_sm_end-vmd_n_ceemdan_sm_start) % 3600) // 60)}m {int((vmd_n_ceemdan_sm_end-vmd_n_ceemdan_sm_start) // 60)}s')
# print(f'\nTIME ELAPSED VMD AND CEEMDAN (ALL MONTHS): {int((vmd_n_ceemdan_end-vmd_n_ceemdan_start) // 3600)}h {int(((vmd_n_ceemdan_end-vmd_n_ceemdan_start) % 3600) // 60)}m {int((vmd_n_ceemdan_end-vmd_n_ceemdan_start) // 60)}s')

# time.sleep(100000)
# decomposed_close_price_IMFs = vmd.fit_transform(signal=close_price_ts)

# close_price_residual = close_price_ts.copy()

# for imf in decomposed_close_price_IMFs:
#     close_price_residual -= imf

# ceemdan = CEEMDAN(trials=200,
#                   epsilon=2.0,
#                   max_imfs=15)

# decomposed_close_price_residual_IMFs = ceemdan.fit_transform(signal=close_price_residual)

# for i, m_1 in enumerate(decomposed_close_price_IMFs[:1]):
#     for j, m_2 in enumerate(decomposed_close_price_IMFs):
#         if i != j:
#             print(f"Dependant: Mode {i}, Cause: Mode {j}")

#             current_pair = np.column_stack([m_1, m_2])
#             result = grangercausalitytests(current_pair, maxlag=30, verbose=True)

#             for lag, test in result.items():
#                 print(f"Lag {lag} — p-value: {test[0]['ssr_ftest'][1]}")

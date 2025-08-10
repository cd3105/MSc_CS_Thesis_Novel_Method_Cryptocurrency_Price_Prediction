import os
import numpy as np
import pandas as pd
from datetime import datetime
from pysdkit import VMD#, MVMD

market_dataset_bp = "All_Crypto_Data/New_Data/Processed/Market_Data/Only_Binance_Market_Data/"
cryptocurrencies = ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']
intervals = ['1_Hour', '2_Hour', '4_Hour', '6_Hour', '8_Hour', '12_Hour', '1_Day']
selected_time_period = pd.date_range(start=datetime(2017, 6, 1), 
                                     end=datetime(2022, 5, 31), 
                                     freq='D')

print(selected_time_period)

def apply_price_decomposition(cc,
                              price_df):
    close_price_ts = np.array(price_df[f'{cc}_CLOSE_PRICE_USD'])
    vmd = VMD(K=11, 
              alpha=500, 
              tau=0.0, 
              tol=1e-9)
    decomposed_close_price_imfs = vmd.fit_transform(signal=close_price_ts)

    for i, m in enumerate(decomposed_close_price_imfs):
        price_df[f'DECOMPOSED_{cc}_CLOSE_PRICE_USD_MODE_{i+1}'] = m

    return price_df

# load new signal
# df = pd.read_csv("All_Crypto_Data/Original_Data/BTC/1_Day/Binance_and_Investing_BTC_Daily_Market_Data_01_06_2017__31_05_2022.csv")

# print(df)

# signal = np.array(df)

# # use variational mode decomposition
# vmd = MVMD(alpha=500, K=11, tau=0.0, tol=1e-9)
# IMFs = vmd.fit_transform(signal=signal)
# print(IMFs.shape)
# print(IMFs[0])
# print(signal.shape)
# vmd.plot_IMFs(save_figure=True, dpi=600)


for cc in cryptocurrencies:
    for iv in intervals[-1:]:
        current_market_df_base_path = f'{market_dataset_bp}/{cc}/{iv}/'
        current_market_df = pd.read_csv(f'{market_dataset_bp}/{cc}/{iv}/{os.listdir(current_market_df_base_path)[0]}')
        current_market_df['TIMESTAMP'] = pd.to_datetime(current_market_df['TIMESTAMP'])
        current_market_df = current_market_df[current_market_df['TIMESTAMP'].isin(selected_time_period)]
        
        #decomposed_close
        current_market_df = apply_price_decomposition(cc, current_market_df)



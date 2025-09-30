import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

experiments = {'Baseline':'All_Crypto_Data/Adopted_Data/Processed/Baseline/',
               'Extended_Binance':'All_Crypto_Data/Adopted_Data/Processed/Extended/',
               'Full_Binance':'All_Crypto_Data/Adopted_Data/Processed/Full/',}

for e in experiments.keys():
    current_price_data_base_path = f'{experiments[e]}Price_Data/Binance/'

    for cc in os.listdir(current_price_data_base_path):
        for f in os.listdir(f"{current_price_data_base_path}{cc}"):
            current_file = os.listdir(f"{current_price_data_base_path}{cc}/{f}/")[0]
            current_df = pd.read_csv(f"{current_price_data_base_path}{cc}/{f}/{current_file}")
            current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])

            plt.figure(figsize=(20, 12))

            plt.plot(current_df['TIMESTAMP'], 
                     current_df[f'{cc}_OPEN_PRICE_USDT'], 
                     label='Open Price', 
                     color='green')
            plt.plot(current_df['TIMESTAMP'], 
                     current_df[f'{cc}_HIGH_PRICE_USDT'], 
                     label='High Price', 
                     color='orange')
            plt.plot(current_df['TIMESTAMP'], 
                     current_df[f'{cc}_LOW_PRICE_USDT'], 
                     label='Low Price', 
                     color='red')
            plt.plot(current_df['TIMESTAMP'], 
                     current_df[f'{cc}_CLOSE_PRICE_USDT'], 
                     label='Close Price', 
                     color='blue')

            plt.title(f'{cc}/USDT OHLC Prices over Time', 
                      fontsize=20, 
                      fontweight='bold')
            plt.xlabel('Time', 
                       fontsize=20, 
                       fontweight='bold')
            plt.ylabel('Price (USDT)', 
                       fontsize=20, 
                       fontweight='bold')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(fontsize=15)
            plt.tight_layout()

            current_plot_path = f"Exploratory_Data_Analysis/Results/Price_Plotting/{e}/{cc}/"

            if not os.path.exists(current_plot_path):
                os.makedirs(current_plot_path)

            if f == '1_Day':
                plt.savefig(f"{current_plot_path}{cc}_Daily_Price_Plot.png")
            elif f == '1_Hour':
                plt.savefig(f"{current_plot_path}{cc}_Hourly_Price_Plot.png")
            else:
                plt.savefig(f"{current_plot_path}{cc}_{f}ly_Price_Plot.png")
                
            plt.close() 

import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

market_data_base_path = "All_Crypto_Data/New_Data/Processed/Market_Data/All_Market_Data/"

for cc in os.listdir(market_data_base_path):
    for f in os.listdir(f"{market_data_base_path}{cc}"):
        periods = ['Initial_Period',
                   'Extended_Period']

        for p in periods:
            current_file = os.listdir(f"{market_data_base_path}{cc}/{f}/")[0]
            current_df = pd.read_csv(f"{market_data_base_path}{cc}/{f}/{current_file}")
            current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])

            if p == 'Initial_Period':
                current_df = current_df[current_df['TIMESTAMP'] <= datetime(2022, 5, 31, 23)]

            plt.figure(figsize=(20, 12))

            plt.plot(current_df['TIMESTAMP'], 
                     current_df[f'{cc}_OPEN_PRICE_USD'], 
                     label='Open Price', 
                     color='green')
            plt.plot(current_df['TIMESTAMP'], 
                     current_df[f'{cc}_HIGH_PRICE_USD'], 
                     label='High Price', 
                     color='orange')
            plt.plot(current_df['TIMESTAMP'], 
                     current_df[f'{cc}_LOW_PRICE_USD'], 
                     label='Low Price', 
                     color='red')
            plt.plot(current_df['TIMESTAMP'], 
                     current_df[f'{cc}_CLOSE_PRICE_USD'], 
                     label='Close Price', 
                     color='blue')

            plt.title(f'{cc} Price in USDT over Time', 
                      fontsize=20, 
                      fontweight='bold')
            plt.xlabel('Time', 
                       fontsize=15, 
                       fontweight='bold')
            plt.ylabel('Price (USDT)', 
                       fontsize=15, 
                       fontweight='bold')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(fontsize=15)
            plt.tight_layout()

            current_plot_path = f"Exploratory_Data_Analysis/Results/Price_Plotting/{cc}/{p}/"

            if not os.path.exists(current_plot_path):
                os.makedirs(current_plot_path)

            if f == '1_Day':
                plt.savefig(f"{current_plot_path}{cc}_Daily_Price_{p}_Plot.png")
            elif f == '1_Hour':
                plt.savefig(f"{current_plot_path}{cc}_Hourly_Price_{p}_Plot.png")
            else:
                plt.savefig(f"{current_plot_path}{cc}_{f}ly_Price_{p}_Plot.png")
            
            plt.close() 

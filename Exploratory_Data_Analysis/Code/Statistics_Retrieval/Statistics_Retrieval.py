import pandas as pd
import os
from datetime import datetime

market_data_base_path = "All_Crypto_Data/New_Data/Processed/Market_Data/All_Market_Data/"

for cc in os.listdir(market_data_base_path):
    for f in os.listdir(f"{market_data_base_path}{cc}"):
        periods = ['Initial_Period',
                   'Extended_Period']

        for p in periods:
            current_file = os.listdir(f"{market_data_base_path}{cc}/{f}/")[0]
            current_df = pd.read_csv(f"{market_data_base_path}{cc}/{f}/{current_file}")
            current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])
            current_results_df = pd.DataFrame(columns=['Observation Count',
                                                       'Mean',
                                                       'Median',
                                                       'Standard Deviation',
                                                       'Min',
                                                       '1%',
                                                       '5%',
                                                       '10%',
                                                       '25%',
                                                       '50%',
                                                       '75%',
                                                       '90%',
                                                       '95%',
                                                       '99%',
                                                       'Max',])

            if p == 'Initial_Period':
                current_df = current_df[current_df['TIMESTAMP'] <= datetime(2022, 5, 31, 23)]
            
            current_results_df.loc[f'{cc} Opening Price ($)'] = [len(current_df[f'{cc}_OPEN_PRICE_USD']), 
                                                                 current_df[f'{cc}_OPEN_PRICE_USD'].mean(), 
                                                                 current_df[f'{cc}_OPEN_PRICE_USD'].median(), 
                                                                 current_df[f'{cc}_OPEN_PRICE_USD'].std(),
                                                                 current_df[f'{cc}_OPEN_PRICE_USD'].min(),
                                                                 current_df[f'{cc}_OPEN_PRICE_USD'].quantile(0.01),
                                                                 current_df[f'{cc}_OPEN_PRICE_USD'].quantile(0.05),
                                                                 current_df[f'{cc}_OPEN_PRICE_USD'].quantile(0.1),
                                                                 current_df[f'{cc}_OPEN_PRICE_USD'].quantile(0.25),
                                                                 current_df[f'{cc}_OPEN_PRICE_USD'].quantile(0.50),
                                                                 current_df[f'{cc}_OPEN_PRICE_USD'].quantile(0.75),
                                                                 current_df[f'{cc}_OPEN_PRICE_USD'].quantile(0.90),
                                                                 current_df[f'{cc}_OPEN_PRICE_USD'].quantile(0.95),
                                                                 current_df[f'{cc}_OPEN_PRICE_USD'].quantile(0.99),
                                                                 current_df[f'{cc}_OPEN_PRICE_USD'].max(),]
            current_results_df.loc[f'{cc} High Price ($)'] = [len(current_df[f'{cc}_HIGH_PRICE_USD']), 
                                                              current_df[f'{cc}_HIGH_PRICE_USD'].mean(), 
                                                              current_df[f'{cc}_HIGH_PRICE_USD'].median(), 
                                                              current_df[f'{cc}_HIGH_PRICE_USD'].std(),
                                                              current_df[f'{cc}_HIGH_PRICE_USD'].min(),
                                                              current_df[f'{cc}_HIGH_PRICE_USD'].quantile(0.01),
                                                              current_df[f'{cc}_HIGH_PRICE_USD'].quantile(0.05),
                                                              current_df[f'{cc}_HIGH_PRICE_USD'].quantile(0.1),
                                                              current_df[f'{cc}_HIGH_PRICE_USD'].quantile(0.25),
                                                              current_df[f'{cc}_HIGH_PRICE_USD'].quantile(0.50),
                                                              current_df[f'{cc}_HIGH_PRICE_USD'].quantile(0.75),
                                                              current_df[f'{cc}_HIGH_PRICE_USD'].quantile(0.90),
                                                              current_df[f'{cc}_HIGH_PRICE_USD'].quantile(0.95),
                                                              current_df[f'{cc}_HIGH_PRICE_USD'].quantile(0.99),
                                                              current_df[f'{cc}_HIGH_PRICE_USD'].max()]
            current_results_df.loc[f'{cc} Low Price ($)'] = [len(current_df[f'{cc}_LOW_PRICE_USD']), 
                                                             current_df[f'{cc}_LOW_PRICE_USD'].mean(), 
                                                             current_df[f'{cc}_LOW_PRICE_USD'].median(), 
                                                             current_df[f'{cc}_LOW_PRICE_USD'].std(),
                                                             current_df[f'{cc}_LOW_PRICE_USD'].min(),
                                                             current_df[f'{cc}_LOW_PRICE_USD'].quantile(0.01),
                                                             current_df[f'{cc}_LOW_PRICE_USD'].quantile(0.05),
                                                             current_df[f'{cc}_LOW_PRICE_USD'].quantile(0.1),
                                                             current_df[f'{cc}_LOW_PRICE_USD'].quantile(0.25),
                                                             current_df[f'{cc}_LOW_PRICE_USD'].quantile(0.50),
                                                             current_df[f'{cc}_LOW_PRICE_USD'].quantile(0.75),
                                                             current_df[f'{cc}_LOW_PRICE_USD'].quantile(0.90),
                                                             current_df[f'{cc}_LOW_PRICE_USD'].quantile(0.95),
                                                             current_df[f'{cc}_LOW_PRICE_USD'].quantile(0.99),
                                                             current_df[f'{cc}_LOW_PRICE_USD'].max()]
            current_results_df.loc[f'{cc} Close Price ($)'] = [len(current_df[f'{cc}_CLOSE_PRICE_USD']), 
                                                               current_df[f'{cc}_CLOSE_PRICE_USD'].mean(), 
                                                               current_df[f'{cc}_CLOSE_PRICE_USD'].median(), 
                                                               current_df[f'{cc}_CLOSE_PRICE_USD'].std(),
                                                               current_df[f'{cc}_CLOSE_PRICE_USD'].min(),
                                                               current_df[f'{cc}_CLOSE_PRICE_USD'].quantile(0.01),
                                                               current_df[f'{cc}_CLOSE_PRICE_USD'].quantile(0.05),
                                                               current_df[f'{cc}_CLOSE_PRICE_USD'].quantile(0.1),
                                                               current_df[f'{cc}_CLOSE_PRICE_USD'].quantile(0.25),
                                                               current_df[f'{cc}_CLOSE_PRICE_USD'].quantile(0.50),
                                                               current_df[f'{cc}_CLOSE_PRICE_USD'].quantile(0.75),
                                                               current_df[f'{cc}_CLOSE_PRICE_USD'].quantile(0.90),
                                                               current_df[f'{cc}_CLOSE_PRICE_USD'].quantile(0.95),
                                                               current_df[f'{cc}_CLOSE_PRICE_USD'].quantile(0.99),
                                                               current_df[f'{cc}_CLOSE_PRICE_USD'].max(),]
            
            current_results_path = f"Exploratory_Data_Analysis/Results/Statistics_Retrieval/{cc}/{p}/"

            if not os.path.exists(current_results_path):
                os.makedirs(current_results_path)

            if f == '1_Day':
                current_results_df.to_csv(f"{current_results_path}Statistics_{p}_{cc}_Daily_Data.csv")
            elif f == '1_Hour':
                current_results_df.to_csv(f"{current_results_path}Statistics_{p}_{cc}_Hourly_Data.csv")
            else:
                current_results_df.to_csv(f"{current_results_path}Statistics_{p}_{cc}_{f}ly_Data.csv")


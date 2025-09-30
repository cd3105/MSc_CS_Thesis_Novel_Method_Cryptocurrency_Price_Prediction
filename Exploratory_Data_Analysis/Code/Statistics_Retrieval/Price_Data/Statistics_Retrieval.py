import pandas as pd
import os
from datetime import datetime

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
            current_results_df = pd.DataFrame(columns=[r'\textbf{O.C.}',
                                                       r'\textbf{Mean}',
                                                       r'\textbf{S.D.}',
                                                       r'\textbf{Min.}',
                                                       r'\textbf{5\%}',
                                                       r'\textbf{25\%}',
                                                       r'\textbf{Median}',
                                                       r'\textbf{75\%}',
                                                       r'\textbf{95\%}',
                                                       r'\textbf{Max.}',])
                
            current_results_df.loc[r'\emph{' + f'{cc}' + r'}/\emph{USDT} OP'] = [f"{int(len(current_df[f'{cc}_OPEN_PRICE_USDT'])):.0f}", 
                                                                                 f"{current_df[f'{cc}_OPEN_PRICE_USDT'].mean():.2f}", 
                                                                                 f"{current_df[f'{cc}_OPEN_PRICE_USDT'].std():.2f}",
                                                                                 f"{current_df[f'{cc}_OPEN_PRICE_USDT'].min():.2f}",
                                                                                 f"{current_df[f'{cc}_OPEN_PRICE_USDT'].quantile(0.05):.2f}",
                                                                                 f"{current_df[f'{cc}_OPEN_PRICE_USDT'].quantile(0.25):.2f}",
                                                                                 f"{current_df[f'{cc}_OPEN_PRICE_USDT'].median():.2f}",
                                                                                 f"{current_df[f'{cc}_OPEN_PRICE_USDT'].quantile(0.75):.2f}",
                                                                                 f"{current_df[f'{cc}_OPEN_PRICE_USDT'].quantile(0.95):.2f}",
                                                                                 f"{current_df[f'{cc}_OPEN_PRICE_USDT'].max():.2f}",]
            current_results_df.loc[r'\emph{' + f'{cc}' + r'}/\emph{USDT} HP'] = [f"{int(len(current_df[f'{cc}_HIGH_PRICE_USDT'])):.0f}", 
                                                                                 f"{current_df[f'{cc}_HIGH_PRICE_USDT'].mean():.2f}", 
                                                                                 f"{current_df[f'{cc}_HIGH_PRICE_USDT'].std():.2f}",
                                                                                 f"{current_df[f'{cc}_HIGH_PRICE_USDT'].min():.2f}",
                                                                                 f"{current_df[f'{cc}_HIGH_PRICE_USDT'].quantile(0.05):.2f}",
                                                                                 f"{current_df[f'{cc}_HIGH_PRICE_USDT'].quantile(0.25):.2f}",
                                                                                 f"{current_df[f'{cc}_HIGH_PRICE_USDT'].median():.2f}",
                                                                                 f"{current_df[f'{cc}_HIGH_PRICE_USDT'].quantile(0.75):.2f}",
                                                                                 f"{current_df[f'{cc}_HIGH_PRICE_USDT'].quantile(0.95):.2f}",
                                                                                 f"{current_df[f'{cc}_HIGH_PRICE_USDT'].max():.2f}",]
            current_results_df.loc[r'\emph{' + f'{cc}' + r'}/\emph{USDT} LP'] = [f"{int(len(current_df[f'{cc}_LOW_PRICE_USDT'])):.0f}", 
                                                                                 f"{current_df[f'{cc}_LOW_PRICE_USDT'].mean():.2f}", 
                                                                                 f"{current_df[f'{cc}_LOW_PRICE_USDT'].std():.2f}",
                                                                                 f"{current_df[f'{cc}_LOW_PRICE_USDT'].min():.2f}",
                                                                                 f"{current_df[f'{cc}_LOW_PRICE_USDT'].quantile(0.05):.2f}",
                                                                                 f"{current_df[f'{cc}_LOW_PRICE_USDT'].quantile(0.25):.2f}",
                                                                                 f"{current_df[f'{cc}_LOW_PRICE_USDT'].median():.2f}",
                                                                                 f"{current_df[f'{cc}_LOW_PRICE_USDT'].quantile(0.75):.2f}",
                                                                                 f"{current_df[f'{cc}_LOW_PRICE_USDT'].quantile(0.95):.2f}",
                                                                                 f"{current_df[f'{cc}_LOW_PRICE_USDT'].max():.2f}",]
            current_results_df.loc[r'\emph{' + f'{cc}' + r'}/\emph{USDT} CP'] = [f"{int(len(current_df[f'{cc}_CLOSE_PRICE_USDT'])):.0f}", 
                                                                                 f"{current_df[f'{cc}_CLOSE_PRICE_USDT'].mean():.2f}", 
                                                                                 f"{current_df[f'{cc}_CLOSE_PRICE_USDT'].std():.2f}",
                                                                                 f"{current_df[f'{cc}_CLOSE_PRICE_USDT'].min():.2f}",
                                                                                 f"{current_df[f'{cc}_CLOSE_PRICE_USDT'].quantile(0.05):.2f}",
                                                                                 f"{current_df[f'{cc}_CLOSE_PRICE_USDT'].quantile(0.25):.2f}",
                                                                                 f"{current_df[f'{cc}_CLOSE_PRICE_USDT'].median():.2f}",
                                                                                 f"{current_df[f'{cc}_CLOSE_PRICE_USDT'].quantile(0.75):.2f}",
                                                                                 f"{current_df[f'{cc}_CLOSE_PRICE_USDT'].quantile(0.95):.2f}",
                                                                                 f"{current_df[f'{cc}_CLOSE_PRICE_USDT'].max():.2f}",]
                
            current_results_path = f"Exploratory_Data_Analysis/Results/Statistics_Retrieval/{e}/{cc}/"

            if not os.path.exists(current_results_path):
                os.makedirs(current_results_path)

            if f == '1_Day':
                current_results_df.to_csv(f"{current_results_path}Statistics_{cc}_Daily_Data.csv")
                current_results_df.to_latex(f"{current_results_path}Statistics_{cc}_Daily_Data.tex")
            elif f == '1_Hour':
                current_results_df.to_csv(f"{current_results_path}Statistics_{cc}_Hourly_Data.csv")
                current_results_df.to_latex(f"{current_results_path}Statistics_{cc}_Hourly_Data.tex")
            else:
                current_results_df.to_csv(f"{current_results_path}Statistics_{cc}_{f}ly_Data.csv")
                current_results_df.to_latex(f"{current_results_path}Statistics_{cc}_{f}ly_Data.tex")


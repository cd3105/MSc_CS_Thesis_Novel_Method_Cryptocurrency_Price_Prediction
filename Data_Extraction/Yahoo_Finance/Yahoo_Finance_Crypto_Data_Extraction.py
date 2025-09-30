import yfinance as yf
import pandas as pd
import os
import time

# Script for retrieving Cryptocurrency Market Data from Yahoo Finance

base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Yahoo_Finance/"
crypto_dict = {}
ticker_retrieval_counter = 0

for cc in os.listdir(base_path):
    current_ticker = f"{cc}-USD"
    cc_key = current_ticker.replace('-', '_')
    
    if len(os.listdir(f"{base_path}{cc}/1_Day")) == 0:
        crypto_dict[cc_key] = yf.Ticker(current_ticker).history(period="max").reset_index()
        
        if len(crypto_dict[cc_key]) == 0:
            ticker_retry = yf.Ticker(current_ticker).history(period="6350d").reset_index(drop=True)

            if 'Date' not in ticker_retry.columns:
                crypto_dict[cc_key] = pd.concat([crypto_dict[cc_key], ticker_retry])
            else:
                crypto_dict[cc_key] = ticker_retry.reset_index(drop=True)
        
        if pd.isna(list(crypto_dict[cc_key]['Date'])[0]) or pd.isna(list(crypto_dict[cc_key]['Date'])[-1]):
            current_file_name = f"Yahoo_Finance_{cc_key}_Daily_UTC_Historical_OHLCV.csv"
        else:
            current_file_name = f"Yahoo_Finance_{cc_key}_Daily_UTC_Historical_OHLCV_{list(crypto_dict[cc_key]['Date'])[0].strftime('%d-%m-%Y').replace('-', '_')}__{list(crypto_dict[cc_key]['Date'])[-1].strftime('%d-%m-%Y').replace('-', '_')}.csv"
        current_df = crypto_dict[cc_key].rename(columns={'Date':'TIMESTAMP',
                                                         'Open':f'{cc}_OPEN_PRICE_USD',
                                                         'High':f'{cc}_HIGH_PRICE_USD',
                                                         'Low':f'{cc}_LOW_PRICE_USD',
                                                         'Close':f'{cc}_CLOSE_PRICE_USD',
                                                         'Volume':f'{cc}_VOLUME_USD',
                                                         'Dividends':f'{cc}_DIVIDENDS_USD',
                                                         'Stock Splits':f'{cc}_STOCK_SPLITS_USD'})
        current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP'])
        current_df.sort_values('TIMESTAMP', ascending=True).to_csv(f"{base_path}{cc}/1_Day/{current_file_name}", index=False)

    ticker_retrieval_counter+=1

    if (ticker_retrieval_counter % 100) == 0:
        print(f"{ticker_retrieval_counter} Tickers Retrieved! Last CC: {cc}")

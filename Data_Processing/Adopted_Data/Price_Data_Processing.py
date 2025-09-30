import pandas as pd
import os
from datetime import datetime

adopted_data_base_path = "All_Crypto_Data/Adopted_Data/"

baseline_experiment = [datetime(2017, 6, 1), datetime(2022, 5, 31, 23)]
extended_experiment = [datetime(2017, 6, 1), datetime(2024, 2, 20, 23)]
full_experiment = [datetime(2017, 6, 1), datetime(2024, 12, 31, 23)]
selected_ccs = ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']

time_spans = {'Baseline':baseline_experiment,
              'Extended':extended_experiment,
              'Full':full_experiment}

sources_daily = {'Binance':['All_Crypto_Data/Crypto_Market_Data/Merged/Binance_Extended/', 'USDT'],
                 'CoinMarketCap':['All_Crypto_Data/Crypto_Market_Data/Merged/CoinMarketCap/', 'USD'],
                 'Investing':['All_Crypto_Data/Crypto_Market_Data/Merged/Investing/', 'USD']}
sources_hourly = {'Binance':['All_Crypto_Data/Crypto_Market_Data/Merged/Binance_via_CoinDesk/', 'USDT']}

for cc in selected_ccs:
    for ts in time_spans:
        for s in sources_daily.keys():
            selected_cols = ['TIMESTAMP',
                             f'{cc}_OPEN_PRICE_{sources_daily[s][1]}',
                             f'{cc}_HIGH_PRICE_{sources_daily[s][1]}',
                             f'{cc}_LOW_PRICE_{sources_daily[s][1]}',
                             f'{cc}_CLOSE_PRICE_{sources_daily[s][1]}']
            
            current_daily_data_path = f'{sources_daily[s][0]}{cc}/1_Day/'
            current_daily_data = pd.read_csv(f'{current_daily_data_path}{os.listdir(f"{current_daily_data_path}")[0]}')[selected_cols]
            current_daily_data['TIMESTAMP'] = pd.to_datetime(current_daily_data['TIMESTAMP'])

            if current_daily_data['TIMESTAMP'].dt.tz is not None:
                current_daily_data['TIMESTAMP'] = current_daily_data['TIMESTAMP'].dt.tz_convert(None)

            current_daily_data = current_daily_data[(current_daily_data['TIMESTAMP'] >= time_spans[ts][0]) & (current_daily_data['TIMESTAMP'] <= time_spans[ts][1])]
            
            current_adopted_data_daily_path_unprocessed = f'{adopted_data_base_path}Unprocessed/{ts}/Price_Data/{s}/{cc}/1_Day/'
            current_adopted_data_daily_path_processed = f'{adopted_data_base_path}Processed/{ts}/Price_Data/{s}/{cc}/1_Day/'

            if not os.path.exists(current_adopted_data_daily_path_unprocessed):
                os.makedirs(current_adopted_data_daily_path_unprocessed)

            if not os.path.exists(current_adopted_data_daily_path_processed):
                os.makedirs(current_adopted_data_daily_path_processed)

            current_daily_data.to_csv(f'{current_adopted_data_daily_path_unprocessed}{cc}_Price_Data.csv',
                                      index=False)

            if (ts == 'Full') and (cc == 'XMR') and (s == 'Binance'):
                fill_data_path = sources_daily['Investing'][0]
                fill_data = pd.read_csv(f'{fill_data_path}{cc}/1_Day/{os.listdir(f"{fill_data_path}{cc}/1_Day/")[0]}')
                fill_data['TIMESTAMP'] = pd.to_datetime(fill_data['TIMESTAMP'])
                fill_data = fill_data[(fill_data['TIMESTAMP'] > datetime(2024, 2, 20, 23)) & (fill_data['TIMESTAMP'] <= time_spans[ts][1])]

                selected_cs = [c for c in fill_data.columns if 'USD' in c]
                new_names_cs = [c.replace('USD', 'USDT') for c in selected_cs]
                fill_data = fill_data.rename(columns=dict(list(zip(selected_cs, new_names_cs))))
                fill_data = fill_data[selected_cols]
                
                pd.concat([current_daily_data, fill_data]).to_csv(f'{current_adopted_data_daily_path_processed}{cc}_Price_Data.csv', 
                                                                  index=False)
            else:
                current_daily_data.to_csv(f'{current_adopted_data_daily_path_processed}{cc}_Price_Data.csv', 
                                          index=False)


for cc in selected_ccs:
    for ts in time_spans:
        for s in sources_hourly.keys():
            selected_cols = ['TIMESTAMP',
                             f'{cc}_OPEN_PRICE_{sources_hourly[s][1]}',
                             f'{cc}_HIGH_PRICE_{sources_hourly[s][1]}',
                             f'{cc}_LOW_PRICE_{sources_hourly[s][1]}',
                             f'{cc}_CLOSE_PRICE_{sources_hourly[s][1]}']
            
            freqs = [f for f in os.listdir(f'{sources_hourly[s][0]}{cc}/') if f != '1_Day']
            
            for f in freqs:
                current_hourly_data_path = f'{sources_hourly[s][0]}{cc}/{f}/'
                current_hourly_data = pd.read_csv(f'{current_hourly_data_path}{os.listdir(f"{current_hourly_data_path}")[0]}')[selected_cols]
                current_hourly_data['TIMESTAMP'] = pd.to_datetime(current_hourly_data['TIMESTAMP'])

                if current_hourly_data['TIMESTAMP'].dt.tz is not None:
                    current_hourly_data['TIMESTAMP'] = current_hourly_data['TIMESTAMP'].dt.tz_convert(None)

                current_hourly_data = current_hourly_data[(current_hourly_data['TIMESTAMP'] >= time_spans[ts][0]) & (current_hourly_data['TIMESTAMP'] <= time_spans[ts][1])]

                current_adopted_data_hourly_path_unprocessed = f'{adopted_data_base_path}Unprocessed/{ts}/Price_Data/{s}/{cc}/1{f}/'
                current_adopted_data_hourly_path_processed = f'{adopted_data_base_path}Processed/{ts}/Price_Data/{s}/{cc}/{f}/'

                if not os.path.exists(current_adopted_data_hourly_path_unprocessed):
                    os.makedirs(current_adopted_data_hourly_path_unprocessed)

                if not os.path.exists(current_adopted_data_hourly_path_processed):
                    os.makedirs(current_adopted_data_hourly_path_processed)

                current_hourly_data.to_csv(f'{current_adopted_data_hourly_path_unprocessed}{cc}_Price_Data.csv', 
                                           index=False)
                current_hourly_data.to_csv(f'{current_adopted_data_hourly_path_processed}{cc}_Price_Data.csv', 
                                           index=False)

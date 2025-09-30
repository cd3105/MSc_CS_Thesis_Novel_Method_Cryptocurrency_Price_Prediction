import pandas as pd
import os
import requests
from io import StringIO
from datetime import datetime

# Script for retrieving Hourly Market Data via the CoinDesk API

selected_ccs_base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Binance_Extended/"

other_ccs_investing_base_path = "All_Crypto_Data/Crypto_Market_Data/Merged/Other_Investing/"

base_paths = [selected_ccs_base_path, other_ccs_investing_base_path]
ccs_viable_data_available = []


for bp in base_paths:
    for cc in [m for m in os.listdir(bp) if not m.endswith('.txt')]:
        current_file = os.listdir(f"{bp}{cc}/1_Day/")[0]
        current_df = pd.read_csv(f"{bp}{cc}/1_Day/{current_file}")
        current_df['TIMESTAMP'] = pd.to_datetime(current_df['TIMESTAMP']).dt.tz_localize(None)

        if (current_df['TIMESTAMP'].min() <= datetime(2017, 6, 1)):
            ccs_viable_data_available.append(cc)


ccs_viable_data_available = sorted(list(set(ccs_viable_data_available)))
selected_ccs = ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']
viable_other_ccs = []
correlation_results_base_path = "Exploratory_Data_Analysis/Results/Correlation_Between_Cryptos/"
significant_p_value_threshold = 0.075

for cc in selected_ccs:
    for data_type in [m for m in os.listdir(f"{correlation_results_base_path}") if (m == '109CCs') or (m == 'YF_CCs') or (m == 'Binance_via_CoinDesk_CCs')]: # Only Data Covering a Large Period of Time until now
        if cc in os.listdir(f"{correlation_results_base_path}{data_type}/Ranked_Correlation/Granger_Causality/"):
            current_granger_causality_results = pd.read_csv(f"{correlation_results_base_path}{data_type}/Ranked_Correlation/Granger_Causality/{cc}/Closing_Price/{cc}_Ranked_by_Mean_Closing_Price_Granger_Causality.csv",
                                                            index_col=0)

            current_significant_granger_causality_results = current_granger_causality_results[current_granger_causality_results[cc] <= significant_p_value_threshold]
            current_significant_granger_causality_results = current_significant_granger_causality_results[current_significant_granger_causality_results.index.isin(ccs_viable_data_available)]
           
            for row in current_significant_granger_causality_results.iterrows():
                if row[0] not in selected_ccs:
                    viable_other_ccs.append(row[0])


hours = [1, 2, 4, 6, 8, 12]
viable_other_ccs = sorted(list(set(viable_other_ccs)))
max_ts_mapping = {}
max_ts = 1747729079

for cc in viable_other_ccs:
    current_response = requests.get('https://data-api.coindesk.com/spot/v1/latest/tick',
                                    params={"market":"binance",
                                            "instruments":f"{cc}-USDT",
                                            "apply_mapping":"true",
                                            "api_key":os.getenv("CD_API_KEY")},
                                    headers={"Content-type":"application/json; charset=UTF-8"}
    )

    current_json = current_response.json()
    current_max_ts = min(max_ts, current_json['Data'][f"{cc}-USDT"]['PRICE_LAST_UPDATE_TS'])
    max_ts_mapping[cc] = current_max_ts


for cc in viable_other_ccs:
    for h in hours:
        current_ts = max_ts_mapping[cc]
        stopping = False
        concat_df = pd.DataFrame()

        print(f"Crypto: {cc}, Hours: {h}, TS: {current_ts}")

        while not stopping:
            current_response = requests.get('https://data-api.coindesk.com/spot/v1/historical/hours',
                                            params={"market":"binance",
                                                    "instrument":f"{cc}-USDT",
                                                    "to_ts": current_ts,
                                                    "limit":int(2000 / h),
                                                    "aggregate":h,
                                                    "fill":"true",
                                                    "apply_mapping":"true",
                                                    "response_format":"CSV",
                                                    "api_key":os.getenv("CD_API_KEY")}, 
                                            headers={"Content-type":"application/json; charset=UTF-8"}
                    )
            
            current_csv_data = StringIO(current_response.text)
            current_df = pd.read_csv(current_csv_data)
            concat_df = pd.concat([current_df, concat_df]).reset_index(drop=True)

            if len(current_df) < int(2000 / h):
                stopping = True
            else:
                current_ts = list(current_df['TIMESTAMP'])[0] - 3600
        
        concat_df = concat_df.drop(['UNIT',
                                    'TYPE',
                                    'MARKET',
                                    'INSTRUMENT',
                                    'MAPPED_INSTRUMENT',
                                    'BASE',
                                    'QUOTE',
                                    'BASE_ID',
                                    'QUOTE_ID',
                                    'TRANSFORM_FUNCTION',
                                    'FIRST_TRADE_TIMESTAMP', 
                                    'LAST_TRADE_TIMESTAMP',
                                    'HIGH_TRADE_TIMESTAMP',
                                    'LOW_TRADE_TIMESTAMP',
                                    'QUOTE_VOLUME_UNKNOWN',
                                    'TOTAL_TRADES_UNKNOWN',
                                    'VOLUME_UNKNOWN'], axis=1).rename(columns={'OPEN':f'{cc}_OPEN_PRICE_USDT',
                                                                               'CLOSE':f'{cc}_CLOSE_PRICE_USDT',
                                                                               'HIGH':f'{cc}_HIGH_PRICE_USDT',
                                                                               'LOW':f'{cc}_LOW_PRICE_USDT',
                                                                               'FIRST_TRADE_PRICE':f'{cc}_FIRST_TRADE_PRICE_USDT',
                                                                               'HIGH_TRADE_PRICE':f'{cc}_HIGH_TRADE_PRICE_USDT',
                                                                               'LOW_TRADE_PRICE':f'{cc}_LOW_TRADE_PRICE_USDT',
                                                                               'LAST_TRADE_PRICE':f'{cc}_LAST_TRADE_PRICE_USDT',
                                                                               'VOLUME':f'{cc}_VOLUME_USDT',
                                                                               'TOTAL_TRADES':f'{cc}_TOTAL_TRADE_COUNT',
                                                                               'TOTAL_TRADES_BUY':f'{cc}_TOTAL_TRADES_BUY',
                                                                               'TOTAL_TRADES_SELL':f'{cc}_TOTAL_TRADES_SELL',
                                                                               'QUOTE_VOLUME':f'{cc}_QUOTE_VOLUME',
                                                                               'VOLUME_BUY':f'{cc}_VOLUME_BUY',
                                                                               'QUOTE_VOLUME_BUY':f'{cc}_QUOTE_VOLUME_BUY',
                                                                               'VOLUME_SELL':f'{cc}_VOLUME_SELL',
                                                                               'QUOTE_VOLUME_SELL':f'{cc}_QUOTE_VOLUME_SELL',})
        concat_df['TIMESTAMP'] = concat_df['TIMESTAMP'].apply(lambda x: datetime.utcfromtimestamp(x))
        current_file_path = f"All_Crypto_Data/Crypto_Market_Data/Merged/Other_Binance_via_CoinDesk/{cc}/{h}_Hour/"

        if h == 1:
            current_file_name = f"Binance_via_CoinDesk_{cc}_USDT_Hourly_UTC_OHLCV_{concat_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{concat_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"
        else:
            current_file_name = f"Binance_via_CoinDesk_{cc}_USDT_{h}_Hourly_UTC_OHLCV_{concat_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{concat_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"
        
        if not os.path.exists(current_file_path):
            os.makedirs(current_file_path)
    
        concat_df.to_csv(f"{current_file_path}{current_file_name}", index=False)


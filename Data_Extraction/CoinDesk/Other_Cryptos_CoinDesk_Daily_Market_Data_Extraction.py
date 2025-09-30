import requests
import pandas as pd
from io import StringIO
from datetime import datetime
import os
from dotenv import load_dotenv

# Script for retrieving Daily Market Data via the CoinDesk API

load_dotenv()

launch_dates_df = pd.read_csv("All_Crypto_Data\Crypto_Launch_Data\Crypto_Launch_Dates.csv")
selected_ccs = ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']
appropriate_ccs = [row['CC'] for idx, row in launch_dates_df.iterrows() if (datetime.strptime(row['LAUNCH_DATE'], "%Y-%m-%d") <= datetime(2017, 6, 1)) and (row['CC'] not in selected_ccs)]
appropriate_binance_ccs = []
max_ts_mapping = {}
max_ts = 1747729079

for cc in appropriate_ccs:
    current_response = requests.get('https://data-api.coindesk.com/spot/v1/latest/tick',
                                    params={"market":"binance",
                                            "instruments":f"{cc}-USDT",
                                            "apply_mapping":"true",
                                            "api_key":os.getenv("CD_API_KEY")},
                                    headers={"Content-type":"application/json; charset=UTF-8"}
    )

    if current_response.status_code == 200:
        current_json = current_response.json()
        current_max_ts = min(max_ts, current_json['Data'][f"{cc}-USDT"]['PRICE_LAST_UPDATE_TS'])

        if cc not in selected_ccs:
            appropriate_binance_ccs.append(cc)
            max_ts_mapping[cc] = current_max_ts


for cc in appropriate_binance_ccs:
    current_ts = max_ts_mapping[cc]
    stopping = False
    concat_df = pd.DataFrame()

    print(f"Crypto: {cc}, TS: {current_ts}")

    while not stopping:
        current_response = requests.get('https://data-api.coindesk.com/spot/v1/historical/days',
                                        params={"market":"binance",
                                                "instrument":f"{cc}-USDT",
                                                "to_ts": current_ts,
                                                "limit":2000,
                                                "aggregate":1,
                                                "fill":"true",
                                                "apply_mapping":"true",
                                                "response_format":"CSV", 
                                                "api_key":os.getenv("CD_API_KEY")}, 
                                        headers={"Content-type":"application/json; charset=UTF-8"}
                                        )
        
        current_csv_data = StringIO(current_response.text)
        current_df = pd.read_csv(current_csv_data)
        concat_df = pd.concat([current_df, concat_df]).reset_index(drop=True)

        if len(current_df) < 2000:
            stopping = True
        else:
            current_ts = list(current_df['TIMESTAMP'])[0] - 86400
        
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
    
    current_file_path = f"All_Crypto_Data/Crypto_Market_Data/Merged/Other_Binance_via_CoinDesk/{cc}/1_Day/"
    current_file_name = f"Binance_via_CoinDesk_{cc}_USDT_Daily_UTC_OHLCV_{concat_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{concat_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"

    if not os.path.exists(current_file_path):
        os.makedirs(current_file_path)
    
    concat_df.to_csv(f"{current_file_path}{current_file_name}", index=False)
    
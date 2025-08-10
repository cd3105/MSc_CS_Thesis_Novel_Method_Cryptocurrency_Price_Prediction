import requests
import pandas as pd
from io import StringIO
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

markets = {'binance': 'Binance_via_CoinDesk',
           'bitfinex': 'Bitfinex_via_CoinDesk',
           'bitstamp': 'Bitstamp_via_CoinDesk',
           'coinbase': 'Coinbase_via_CoinDesk',
           'kraken': 'Kraken_via_CoinDesk',
           'okcoin': 'OKCoin_via_CoinDesk',
           'poloniex': 'Poloniex_via_CoinDesk',
           'upbit': 'Upbit_via_CoinDesk'}

currency_pair_half = {'binance': '-USDT',
                      'bitfinex': '-USD',
                      'bitstamp': '-USD',
                      'coinbase': '-USD',
                      'kraken': '-USD',
                      'okcoin': '-USD',
                      'poloniex': '-USDT',
                      'upbit': '-USDT'}

cryptocurrencies_ts = {'binance': {'BTC': 1747729079, 'ETH': 1747729079, 'LTC': 1747729079, 'XRP': 1747729079, 'XMR': 1708387200},
                       'bitfinex': {'BTC': 1747729079, 'ETH': 1747729079, 'LTC': 1747729079, 'XRP': 1747729079, 'XMR': 1747729079},
                       'bitstamp': {'BTC': 1747729079, 'ETH': 1747729079, 'LTC': 1747729079, 'XRP': 1747729079},
                       'coinbase': {'BTC': 1747729079, 'ETH': 1747729079, 'LTC': 1747729079, 'XRP': 1747729079},
                       'kraken': {'BTC': 1747729079, 'ETH': 1747729079, 'LTC': 1747729079, 'XRP': 1747729079, 'XMR': 1747729079},
                       'okcoin': {'BTC': 1747729079, 'ETH': 1747729079, 'LTC': 1670544000},
                       'poloniex': {'BTC': 1747729079, 'ETH': 1747729079, 'LTC': 1747729079, 'XRP': 1747729079, 'XMR': 1747729079},
                       'upbit': {'BTC': 1747729079, 'ETH': 1747729079, 'LTC': 1747729079, 'XRP': 1747729079, 'XMR': 1571875200}}

cryptocurrencies_to_retrieve = ['BTC', 'ETH', 'LTC', 'XRP', 'XMR']

for cc in cryptocurrencies_to_retrieve:
    for mk in markets.keys():
        if cc in cryptocurrencies_ts[mk].keys():
            current_ts = cryptocurrencies_ts[mk][cc]
            stopping = False
            concat_df = pd.DataFrame()

            print(f"Crypto: {cc}, Market: {mk}, TS: {current_ts}")

            while not stopping:
                current_response = requests.get('https://data-api.coindesk.com/spot/v1/historical/days',
                                                params={"market":mk,
                                                        "instrument":cc + currency_pair_half[mk],
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

            current_file_path = f"All_Crypto_Data/Crypto_Market_Data/Merged/{markets[mk]}/{cc}/1_Day/"
            current_file_name = f"{markets[mk]}_{cc}{currency_pair_half[mk].replace('-', '_')}_Daily_UTC_OHLCV_{concat_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{concat_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv"
            concat_df.to_csv(f"{current_file_path}{current_file_name}", index=False)

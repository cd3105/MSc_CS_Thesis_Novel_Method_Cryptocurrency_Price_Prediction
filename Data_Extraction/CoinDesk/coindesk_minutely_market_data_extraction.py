import requests
import pandas as pd
from io import StringIO
from datetime import datetime

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

cryptocurrencies_ts = {'binance': {'BTC': 1747747484, 'ETH': 1747747484, 'LTC': 1747747484, 'XRP': 1747747484, 'XMR': 1708397940},
                       'bitfinex': {'BTC': 1747747484, 'ETH': 1747747484, 'LTC': 1747747484, 'XRP': 1747747484, 'XMR': 1747747484},
                       'bitstamp': {'BTC': 1747747484, 'ETH': 1747747484, 'LTC': 1747747484, 'XRP': 1747747484},
                       'coinbase': {'BTC': 1747747484, 'ETH': 1747747484, 'LTC': 1747747484, 'XRP': 1747747484},
                       'kraken': {'BTC': 1747747484, 'ETH': 1747747484, 'LTC': 1747747484, 'XRP': 1747747484, 'XMR': 1747747484},
                       'okcoin': {'BTC': 1747747484, 'ETH': 1747747484, 'LTC': 1670564460},
                       'poloniex': {'BTC': 1747747484, 'ETH': 1747747484, 'LTC': 1747747484, 'XRP': 1747747484, 'XMR': 1747747484},
                       'upbit': {'BTC': 1747747484, 'ETH': 1747747484, 'LTC': 1747747484, 'XRP': 1747747484, 'XMR': 1571956980}}

cryptocurrencies_to_retrieve = ['BTC', 'ETH', 'LTC', 'XRP', 'XMR']

for cc in cryptocurrencies_to_retrieve:
    for mk in markets.keys():
        if cc in cryptocurrencies_ts[mk].keys():
            current_ts = cryptocurrencies_ts[mk][cc]
            stopping = False
            call_counter = 0
            concat_df = pd.DataFrame()

            print(f"Crypto: {cc}, Market: {mk}, TS: {current_ts}")

            while not stopping:
                current_response = requests.get('https://data-api.coindesk.com/spot/v1/historical/minutes',
                                                params={"market":mk,
                                                        "instrument":cc + currency_pair_half[mk],
                                                        "to_ts": current_ts,
                                                        "limit":2000,
                                                        "aggregate":1,
                                                        "fill":"true",
                                                        "apply_mapping":"true",
                                                        "response_format":"CSV",
                                                        "api_key":"c0ef8e605cdb2f14eb04ab7b24c921f4cf8170286873a3cc8ad285d1f6667966"},
                                                headers={"Content-type":"application/json; charset=UTF-8"}
                )

                current_csv_data = StringIO(current_response.text)
                current_df = pd.read_csv(current_csv_data)
                concat_df = pd.concat([current_df, concat_df]).reset_index(drop=True)
                call_counter+=1

                if (call_counter % 500) == 0:
                    print(f"Call: {call_counter}")

                if len(current_df) < 2000:
                    stopping = True
                else:
                    current_ts = list(current_df['TIMESTAMP'])[0] - 60

            current_file_path = f"All_Crypto_Data/Crypto_Market_Data/{markets[mk]}/{cc}/1_Minute/"
            current_file_name = f"{markets[mk]}_{cc}{currency_pair_half[mk].replace('-', '_')}_Minutely_UTC_OHLCV_{datetime.utcfromtimestamp(int(list(concat_df['TIMESTAMP'])[0])).strftime('%d-%m-%Y')}__{datetime.utcfromtimestamp(int(list(concat_df['TIMESTAMP'])[-1])).strftime('%d-%m-%Y')}.csv"
            concat_df.to_csv(current_file_path + current_file_name)

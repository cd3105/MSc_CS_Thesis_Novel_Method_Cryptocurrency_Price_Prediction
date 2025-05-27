import nasdaqdatalink

data_code_dict = {"ADAUSD": "ADA_USD_OHLCV",
                  "AMPUSD": "AMP_USD_OHLCV",
                  "AVAXUSD": "AVAX_USD_OHLCV",
                  "BCHUSD": "BCH_USD_OHLCV",
                  "BTCUSD": "BTC_USD_OHLCV",
                  "DOGEUSD": "DOGE_USD_OHLCV",
                  "DOTUSD": "DOT_USD_OHLCV",
                  "EOSUSD": "EOS_USD_OHLCV",
                  "ETHUSD": "ETH_USD_OHLCV",
                  "LINKUSD": "LINK_USD_OHLCV",
                  "LTCUSD": "LTC_USD_OHLCV",
                  "LUNAUSD": "LUNA_USD_OHLCV",
                  "MATICUSD": "MATIC_USD_OHLCV",
                  "SOLUSD": "SOL_USD_OHLCV",
                  "TRXUSD": "TRX_USD_OHLCV",
                  "XLMUSD": "XLM_USD_OHLCV",
                  "XMRUSD": "XMR_USD_OHLCV",
                  "ZECUSD": "ZEC_USD_OHLCV",}

nasdaqdatalink.ApiConfig.api_key = "EMdb_Jf-aEGJU3xsUZEi"

for k in list(data_code_dict.keys()):
    print(k)

    current_df = nasdaqdatalink.get_table('QDL/BITFINEX',code=k).drop(['code'], axis=1).rename(columns={'date':'Time'}).sort_values(by='Time').reset_index(drop=True)
    current_file_name = f"Blockchain_via_Quandl_BTC_USD_Daily_{data_code_dict[k]}_{list(current_df['Time'])[0].strftime('%d-%m-%Y').replace('-', '_')}__{list(current_df['Time'])[-1].strftime('%d-%m-%Y').replace('-', '_')}.csv"
    current_file_path = f"All_Crypto_Data/Crypto_Market_Data/Bitfinex_via_Quandl/{k.removesuffix('USD')}/1_Day/"

    current_df.to_csv(current_file_path + current_file_name)


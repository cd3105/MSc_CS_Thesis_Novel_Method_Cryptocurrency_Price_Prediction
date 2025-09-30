import nasdaqdatalink
import os
from dotenv import load_dotenv

# Script for extracting Bitfinex Data using the NASDAQ API

load_dotenv()

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

nasdaqdatalink.ApiConfig.api_key = os.getenv("NASDAQ_API_KEY")

for k in list(data_code_dict.keys()):
    current_df = nasdaqdatalink.get_table('QDL/BITFINEX',code=k).drop(['code'], axis=1).rename(columns={'date':'TIMESTAMP'}).sort_values(by='TIMESTAMP').reset_index(drop=True)
    current_file_name = f"Blockchain_via_Quandl_BTC_USD_Daily_{data_code_dict[k]}_{list(current_df['TIMESTAMP'])[0].strftime('%d-%m-%Y').replace('-', '_')}__{list(current_df['TIMESTAMP'])[-1].strftime('%d-%m-%Y').replace('-', '_')}.csv"
    current_file_path = f"All_Crypto_Data/Crypto_Market_Data/Bitfinex_via_Quandl/{k.removesuffix('USD')}/1_Day/"

    current_df.rename(columns={'high':f"{k.removesuffix('USD')}_HIGH_PRICE_USD",
                               'low':f"{k.removesuffix('USD')}_LOW_PRICE_USD",
                               'mid':f"{k.removesuffix('USD')}_MID_PRICE_USD",
                               'last':f"{k.removesuffix('USD')}_LAST_PRICE_USD",
                               'bid':f"{k.removesuffix('USD')}_BID_PRICE_USD",
                               'ask':f"{k.removesuffix('USD')}_ASK_PRICE_USD",
                               'volume':f"{k.removesuffix('USD')}_VOLUME_USD",}).to_csv(current_file_path + current_file_name, index=False)


import yfinance as yf

# Load the stock & Get historical prices

crypto_dict = {'BTC_USD':yf.Ticker("BTC-USD").history(period="max").reset_index(),
               'ETH_USD':yf.Ticker("ETH-USD").history(period="max").reset_index(),
               'XRP_USD': yf.Ticker("XRP-USD").history(period="max").reset_index(),
               'LTC_USD': yf.Ticker("LTC-USD").history(period="max").reset_index(),
               'XMR_USD': yf.Ticker("XMR-USD").history(period="max").reset_index()}

for cc in crypto_dict.keys():
    file_path = f"All_Crypto_Data/Crypto_Market_Data/Yahoo_Finance/{cc.split('_')[0]}/1_Day/"
    file_name = f"Yahoo_Finance_{cc}_Daily_UTC_Historical_OHLCV_{list(crypto_dict[cc]['Date'])[0].strftime('%d-%m-%Y').replace('-', '_')}__{list(crypto_dict[cc]['Date'])[-1].strftime('%d-%m-%Y').replace('-', '_')}.csv"
    crypto_dict[cc].to_csv(file_path + file_name)

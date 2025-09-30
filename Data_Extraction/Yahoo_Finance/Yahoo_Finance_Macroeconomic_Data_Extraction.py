import yfinance as yf

# Script for extracting Macroeconomic Indicators from Yahoo Finance

macroeconomic_dict = {'EUR_USD':yf.Ticker("EURUSD=X").history(period="max").reset_index(),
                      'CNY_USD':yf.Ticker("CNYUSD=X").history(period="max").reset_index(),
                      'CHF_USD':yf.Ticker("CHFUSD=X").history(period="max").reset_index(),
                      'JPY_USD':yf.Ticker("JPYUSD=X").history(period="max").reset_index(),
                      'GBP_USD':yf.Ticker("GBPUSD=X").history(period="max").reset_index(),
                      'AUD_USD': yf.Ticker("AUDUSD=X").history(period="max").reset_index(),
                      'CAD_USD': yf.Ticker("CADUSD=X").history(period="max").reset_index(),
                      'DKK_USD': yf.Ticker("DKKUSD=X").history(period="max").reset_index(),
                      'NZD_USD': yf.Ticker("NZDUSD=X").history(period="max").reset_index(),
                      'NOK_USD': yf.Ticker("NOKUSD=X").history(period="max").reset_index(),
                      'SEK_USD': yf.Ticker("SEKUSD=X").history(period="max").reset_index(),
                      'SGD_USD': yf.Ticker("SGDUSD=X").history(period="max").reset_index(),
                      'RUB_USD': yf.Ticker("RUBUSD=X").history(period="max").reset_index(),
                      'S&P500_Index_USD':yf.Ticker("^GSPC").history(period="max").reset_index(),
                      'Eurostoxx50_Index_EUR':yf.Ticker("^STOXX50E").history(period="max").reset_index(),
                      'DJI_Index_USD':yf.Ticker("^DJI").history(period="max").reset_index(),
                      'NASDAQ_Index_USD':yf.Ticker("^IXIC").history(period="max").reset_index(),
                      'NASDAQ100_Index_USD': yf.Ticker("^NDX").history(period="max").reset_index(),
                      'SSE_Index_CNY':yf.Ticker("000001.SS").history(period="max").reset_index(),
                      'VIX_Index_USD':yf.Ticker("^VIX").history(period="max").reset_index(),
                      'Nikkei225_Index_JPY':yf.Ticker("^N225").history(period="max").reset_index(),
                      'CAC40_Index_EUR': yf.Ticker("^FCHI").history(period="max").reset_index(),
                      'CSI300_Index_CNY': yf.Ticker("000300.SS").history(period="max").reset_index(),
                      'FTSE_Index_GBP':yf.Ticker("^FTSE").history(period="max").reset_index(),
                      'NYSE_Index_USD':yf.Ticker("^NYA").history(period="max").reset_index(),
                      'HSCE_Index_HKD': yf.Ticker("^HSCE").history(period="max").reset_index(),
                      'HSI_Index_HKD': yf.Ticker("^HSI").history(period="max").reset_index(),
                      'KOSPI_Index_KRW': yf.Ticker("^KS11").history(period="max").reset_index(),
                      'STI_Index_SGD': yf.Ticker("^STI").history(period="max").reset_index(),
                      'GVZ_Index_USD': yf.Ticker("^GVZ").history(period="max").reset_index(),
                      'Interest_Rate_10_Years_USD': yf.Ticker("^TNX").history(period="max").reset_index(),
                      'Treasury_Bond_Futures_USD': yf.Ticker("ZB=F").history(period="max").reset_index(),
                      'Treasury_Yield_5_Years_USD': yf.Ticker("^FVX").history(period="max").reset_index(),
                      'Treasury_Yield_30_Years_USD': yf.Ticker("^TYX").history(period="max").reset_index(),
                      'USD_Index': yf.Ticker("DX-Y.NYB").history(period="max").reset_index(),
                      'Gold_Futures_USD':yf.Ticker("GC=F").history(period="max").reset_index(),
                      'Silver_Futures_USD': yf.Ticker("SI=F").history(period="max").reset_index(),
                      'Copper_Futures_USD': yf.Ticker("HG=F").history(period="max").reset_index(),
                      'Crude_Oil_Futures_USD':yf.Ticker("CL=F").history(period="max").reset_index(),
                      'Brent_Oil_Futures_USD': yf.Ticker("BZ=F").history(period="max").reset_index(),
                      'Natural_Gas_Futures_USD': yf.Ticker("NG=F").history(period="max").reset_index(),}

for me in macroeconomic_dict.keys():
    file_path = f"All_Crypto_Data/Macroeconomic_Data/Unmerged/Yahoo_Finance/1_Day/"
    file_name = f"Yahoo_Finance_{me}_Daily_UTC_Historical_OHLCV_{list(macroeconomic_dict[me]['Date'])[0].strftime('%d-%m-%Y').replace('-', '_')}__{list(macroeconomic_dict[me]['Date'])[-1].strftime('%d-%m-%Y').replace('-', '_')}.csv"
    macroeconomic_dict[me].to_csv(file_path + file_name)

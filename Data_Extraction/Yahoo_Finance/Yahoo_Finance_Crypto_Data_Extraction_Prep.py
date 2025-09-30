import os
import requests
import yfinance as yf
from bs4 import BeautifulSoup

# Script to allow Cryptocurrency Market Data retrieval from Yahoo Finance

all_ccs = []
crypto_dict = {}
headers = {"User-Agent": "Mozilla/5.0"}

for s in range(0, 9801, 100):
    current_url = f"https://finance.yahoo.com/markets/crypto/all/?start={s}&count=100"
    current_page = requests.get(current_url, headers=headers)
    current_soup = BeautifulSoup(current_page.content, "html.parser")
    current_symbol_spans = current_soup.findAll('span', class_='symbol yf-1jsynna')

    for ss in current_symbol_spans:
        current_cc = ss.text[:-5]
        map_path = f"All_Crypto_Data/Crypto_Market_Data/Merged/Yahoo_Finance/{current_cc}/1_Day/"

        if not os.path.exists(map_path):
            os.makedirs(map_path, exist_ok=True)  

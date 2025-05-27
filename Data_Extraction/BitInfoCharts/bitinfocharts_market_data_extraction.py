import re
import ast
from datetime import datetime
import requests
import pandas as pd
from bs4 import BeautifulSoup

url_dict = {'Market_Cap': "https://bitinfocharts.com/comparison/marketcap-btc-eth-xrp-doge-ada-ltc-bch-xmr-dash-zec.html#alltime",
            'Market_Cap_2': "https://bitinfocharts.com/comparison/marketcap-sol-trx-eos-atom-xlm-iot-iota.html#alltime",
            'Average_Price_USD': "https://bitinfocharts.com/comparison/price-btc-usdt-eth-sol-xrp-doge-ada-ltc-bch-xmr.html#alltime",
            'Average_Price_USD_2': "https://bitinfocharts.com/comparison/price-trx-bnb-link-dot-avax-eos-atom-xlm-dash-amp.html#alltime",
            'Average_Price_USD_3': "https://bitinfocharts.com/comparison/price-zec-xem-luna-matic.html#alltime",}
coin_mapping_dict = {'Market_Cap': ['BTC', 'ETH', 'XRP', 'DOGE', 'ADA', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Market_Cap_2': ['SOL', 'TRX', 'EOS', 'ATOM', 'XLM', 'IOTA'],
                     'Average_Price_USD': ['BTC', 'USDT', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'LTC', 'BCH', 'XMR'],
                     'Average_Price_USD_2': ['TRX', 'BNB', 'LINK', 'DOT', 'AVAX', 'EOS', 'ATOM', 'XLM', 'DASH', 'AMP'],
                     'Average_Price_USD_3': ['ZEC', 'NEM', 'LUNA', 'MATIC']}
file_path = 'All_Crypto_Data/Crypto_Market_Data/BitInfoCharts/1_Day/'

for k in url_dict.keys():
    current_url = url_dict[k]
    current_coin_mapping = coin_mapping_dict[k]
    current_page = requests.get(current_url)
    current_soup = BeautifulSoup(current_page.content, "html.parser")
    current_body = current_soup.find('body')
    current_required_script = current_body.find_all('script')[-3].text

    current_graph_data = re.search(r'Dygraph\(.*?,\s*(\[\[.*?\]\])', current_required_script, re.DOTALL).group(1)
    current_preprocessed_graph_data = re.sub(r'new Date\("([\d/]+)"\)', r'"\1"', current_graph_data)
    current_preprocessed_graph_data = current_preprocessed_graph_data.replace('/', '-')
    current_preprocessed_graph_data = ast.literal_eval(current_preprocessed_graph_data.replace('null', 'None'))

    current_df = pd.DataFrame(current_preprocessed_graph_data)
    current_df.columns = ['Time'] + [c + f'_{k}' for c in current_coin_mapping]

    current_file_name = f"BitInfoCharts_Daily_{k}_{'_'.join(current_coin_mapping)}_{datetime.strptime(list(current_df['Time'])[0], '%Y-%m-%d').strftime('%d-%m-%Y').replace('-', '_')}__{datetime.strptime(list(current_df['Time'])[-1], '%Y-%m-%d').strftime('%d-%m-%Y').replace('-', '_')}.csv"
    current_df.to_csv(file_path + current_file_name)

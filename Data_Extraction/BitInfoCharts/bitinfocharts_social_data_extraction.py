import re
import ast
from datetime import datetime
import requests
import pandas as pd
from bs4 import BeautifulSoup

url_dict = {'Tweet_Volume_Full_Name_Hashtag': "https://bitinfocharts.com/comparison/tweets-btc-eth-sol-xrp-doge-ada-bnb-ltc-xmr.html#alltime",
            'Tweet_Volume_Full_Name_Hashtag_2': "https://bitinfocharts.com/comparison/tweets-trx-link-dot-avax-bch-eos-atom-xlm-dash-zec.html#alltime",
            'Tweet_Volume_Full_Name_Hashtag_3': "https://bitinfocharts.com/comparison/tweets-avax-xem-luna-amp-matic.html#alltime",
            'Google_Trends_Full_Name': "https://bitinfocharts.com/comparison/google_trends-btc-eth-doge-ltc-xmr.html#alltime"}
coin_mapping_dict = {'Tweet_Volume_Full_Name_Hashtag': ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'BNB', 'LTC', 'XMR'],
                     'Tweet_Volume_Full_Name_Hashtag_2': ['TRX', 'LINK', 'DOT', 'AVAX', 'BCH', 'EOS', 'ATOM', 'XLM', 'DASH', 'ZEC'],
                     'Tweet_Volume_Full_Name_Hashtag_3': ['AVAX', 'NEM', 'LUNA', 'AMP', 'MATIC'],
                     'Google_Trends_Full_Name': ['BTC', 'ETH', 'DOGE', 'LTC', 'XMR']}
file_path_dict = {'Tweet_Volume_Full_Name_Hashtag': 'All_Crypto_Data/Crypto_Twitter_Data/BitInfoCharts/1_Day/',
                  'Tweet_Volume_Full_Name_Hashtag_2': 'All_Crypto_Data/Crypto_Twitter_Data/BitInfoCharts/1_Day/',
                  'Tweet_Volume_Full_Name_Hashtag_3': 'All_Crypto_Data/Crypto_Twitter_Data/BitInfoCharts/1_Day/',
                  'Google_Trends_Full_Name': "All_Crypto_Data/Crypto_Google_Trends/Merged/Google_Trends_via_BitInfoCharts/1_Day/"}

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
    current_df.to_csv(file_path_dict[k] + current_file_name)

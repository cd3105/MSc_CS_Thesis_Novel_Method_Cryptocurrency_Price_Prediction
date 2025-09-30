import re
import ast
from datetime import datetime
import requests
import pandas as pd
from bs4 import BeautifulSoup

# Script for retrieving Daily Blockchain Data related to several cryptocurrencies

url_dict = {'Confirmed_Transaction_Count': "https://bitinfocharts.com/comparison/transactions-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime",
            'Average_Block_Size_Bytes': "https://bitinfocharts.com/comparison/size-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime",
            'Sent_From_Unique_Address_Count': "https://bitinfocharts.com/comparison/sentbyaddress-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime",
            'Average_Mining_Difficulty_Hashes': "https://bitinfocharts.com/comparison/difficulty-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime",
            'Average_Hash_Rate_Hashes_Per_Second': "https://bitinfocharts.com/comparison/hashrate-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime",
            'Mining_Profitability': "https://bitinfocharts.com/comparison/mining_profitability-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime",
            'Sent_Coins_USD': "https://bitinfocharts.com/comparison/sentinusd-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime",
            'Average_Transaction_Fee_USD': "https://bitinfocharts.com/comparison/transactionfees-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime",
            'Median_Transaction_Fee_USD': "https://bitinfocharts.com/comparison/median_transaction_fee-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime",
            'Average_Block_Time_Minutes': "https://bitinfocharts.com/comparison/confirmationtime-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime",
            'Average_Transaction_Value_USD': "https://bitinfocharts.com/comparison/transactionvalue-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime",
            'Median_Transaction_Value_USD': "https://bitinfocharts.com/comparison/mediantransactionvalue-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime",
            'Active_Address_Count': "https://bitinfocharts.com/comparison/activeaddresses-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime",
            'Top_100_Richest_Percentage_of_Total_Coins': "https://bitinfocharts.com/comparison/top100cap-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime",
            'Average_Fee_Percentage_in_Total_Block_Reward': "https://bitinfocharts.com/comparison/fee_to_reward-btc-eth-xrp-doge-ltc-bch-xmr-dash-zec.html#alltime"}
coin_mapping_dict = {'Confirmed_Transaction_Count': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Average_Block_Size_Bytes': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Sent_From_Unique_Address_Count': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Average_Mining_Difficulty_Hashes': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Average_Hash_Rate_Hashes_Per_Second': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Mining_Profitability': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Sent_Coins_USD': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Average_Transaction_Fee_USD': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Median_Transaction_Fee_USD': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Average_Block_Time_Minutes': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Average_Transaction_Value_USD': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Median_Transaction_Value_USD': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Active_Address_Count': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Top_100_Richest_Percentage_of_Total_Coins': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC'],
                     'Average_Fee_Percentage_in_Total_Block_Reward': ['BTC', 'ETH', 'XRP', 'DOGE', 'LTC', 'BCH', 'XMR', 'DASH', 'ZEC']}
file_path = 'All_Crypto_Data/Blockchain_Data/Unmerged/BitInfoCharts/1_Day/'

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

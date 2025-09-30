import requests
import pandas as pd
import json

# Script for extracting the Fear and Greed Index using the Alternative API

url = "https://api.alternative.me/fng/?limit=0&format=json&date_format=world"
response = requests.get(url)

response_json = json.loads(response.text)
f_n_g_df = pd.DataFrame(response_json["data"]).rename(columns={'timestamp':'TIMESTAMP', 
                                                               'value':'FEAR_AND_GREED_INDEX', 
                                                               'value_classification':'FEAR_AND_GREED_INDEX_CLASSIFICATION'}).drop(['time_until_update'], 
                                                                                                                                   axis=1)[['TIMESTAMP', 
                                                                                                                                            'FEAR_AND_GREED_INDEX', 
                                                                                                                                            'FEAR_AND_GREED_INDEX_CLASSIFICATION']]
f_n_g_df['TIMESTAMP'] = pd.to_datetime(f_n_g_df['TIMESTAMP'], 
                                       format='%d-%m-%Y')
f_n_g_df = f_n_g_df.sort_values('TIMESTAMP').reset_index(drop=True)

f_n_g_df.to_csv(f"All_Crypto_Data/Crypto_F_and_G_Index/Alternative/1_Day/Alternative_Daily_Fear_and_Greed_Index_{list(f_n_g_df['TIMESTAMP'])[0].strftime('%d_%m_%Y')}__{list(f_n_g_df['TIMESTAMP'])[-1].strftime('%d_%m_%Y')}.csv", 
                index=False)

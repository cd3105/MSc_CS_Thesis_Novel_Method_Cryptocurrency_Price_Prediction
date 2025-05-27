import requests
import pandas as pd
import json

url = "https://api.alternative.me/fng/?limit=0&format=json&date_format=world"
response = requests.get(url)

response_json = json.loads(response.text)
f_n_g_df = pd.DataFrame(response_json["data"]).rename(columns={'timestamp':'Time', 'value':'Fear_and_Greed_Index', 'value_classification':'Fear_and_Greed_Index_Classification'})
f_n_g_df['Time'] = pd.to_datetime(f_n_g_df['Time'], format='%d-%m-%Y')
f_n_g_df = f_n_g_df.sort_values('Time').reset_index(drop=True)

f_n_g_df.to_csv(f"All_Crypto_Data/Crypto_F_and_G_Index/Alternative/1_Day/Alternative_Daily_Fear_and_Greed_Index_{list(f_n_g_df['Time'])[0].strftime('%d-%m-%Y').replace('-','_')}__{list(f_n_g_df['Time'])[-1].strftime('%d-%m-%Y').replace('-','_')}.csv")

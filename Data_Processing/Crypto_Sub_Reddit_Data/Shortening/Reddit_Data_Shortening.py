import pandas as pd
import os
from datetime import datetime

raw_reddit_data_base_path = "All_Crypto_Data/Crypto_Sub_Reddit_Data/Raw_Sub_Reddit_Text_Data/"
preprocessed_reddit_data_base_path = "All_Crypto_Data/Crypto_Sub_Reddit_Data/Preprocessed_Sub_Reddit_Text_Data/"

new_shortened_raw_reddit_data_base_path = "All_Crypto_Data/Crypto_Sub_Reddit_Data/Shortened_Raw_Sub_Reddit_Text_Data/"
new_shortened_preprocessed_reddit_data_base_path = "All_Crypto_Data/Crypto_Sub_Reddit_Data/Shortened_Preprocessed_Sub_Reddit_Text_Data/"

start_date = datetime(2017, 6, 1)

print("\n Raw Data:\n")

for cc in os.listdir(raw_reddit_data_base_path):
    print(f"\t- CC: {cc}")

    for sr_df in os.listdir(f"{raw_reddit_data_base_path}{cc}/Extracted_Reddit_Data/"):
        print(f"\t\t- SR DF: {sr_df}")

        current_raw_data_df = pd.read_csv(f"{raw_reddit_data_base_path}{cc}/Extracted_Reddit_Data/{sr_df}")
        current_raw_data_df['TIMESTAMP'] = pd.to_datetime(current_raw_data_df['TIMESTAMP'], 
                                                          format="%H:%M %d-%m-%Y")
        current_shortened_raw_data_df = current_raw_data_df[current_raw_data_df['TIMESTAMP'] >= start_date].reset_index(drop=True)
        current_shortened_raw_data_path = f"{new_shortened_raw_reddit_data_base_path}{cc}/"

        if not os.path.exists(current_shortened_raw_data_path):
            os.makedirs(current_shortened_raw_data_path)
        
        current_shortened_raw_data_df.to_csv(f"{current_shortened_raw_data_path}{sr_df}", 
                                             index=False)

print("\nPreprocessed Data:\n")

for pm in os.listdir(preprocessed_reddit_data_base_path):
    print(f"\t-PM: {pm}")

    for cc in os.listdir(f"{preprocessed_reddit_data_base_path}{pm}/"):
        print(f"\t\t- CC: {cc}")

        for sr_df in os.listdir(f"{preprocessed_reddit_data_base_path}{pm}/{cc}/"):
            print(f"\t\t\t- SR DF: {sr_df}")

            current_preprocessed_data_df = pd.read_csv(f"{preprocessed_reddit_data_base_path}{pm}/{cc}/{sr_df}")
            current_preprocessed_data_df['TIMESTAMP'] = pd.to_datetime(current_preprocessed_data_df['TIMESTAMP'], 
                                                                       format="%Y-%m-%d %H:%M:%S")
            current_shortened_preprocessed_data_df = current_preprocessed_data_df[current_preprocessed_data_df['TIMESTAMP'] >= start_date]
            current_shortened_preprocessed_data_path = f"{new_shortened_preprocessed_reddit_data_base_path}{pm}/{cc}/"
            
            if not os.path.exists(current_shortened_preprocessed_data_path):
                os.makedirs(current_shortened_preprocessed_data_path)
            
            current_shortened_preprocessed_data_df.to_csv(f"{current_shortened_preprocessed_data_path}{sr_df}", 
                                                          index=False)

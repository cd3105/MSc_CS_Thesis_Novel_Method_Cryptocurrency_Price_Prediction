import pandas as pd 
import os

selected_ccs = ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']

for cc in selected_ccs:
    current_google_trends_data_fp = f"All_Crypto_Data/Crypto_Google_Trends/Merged/Google_Trends_via_BitInfoCharts/{cc}/1_Day/"
    current_twitter_data_fp = f"All_Crypto_Data/Crypto_Twitter_Data/BitInfoCharts/{cc}/1_Day/"
    other_data_df = pd.DataFrame(columns=["TIMESTAMP"])

    if os.path.exists(current_google_trends_data_fp):
        current_google_trends_data_fn = os.listdir(current_google_trends_data_fp)[0]
        current_google_trends_data_df = pd.read_csv(f"{current_google_trends_data_fp}{current_google_trends_data_fn}")
        
        other_data_df = pd.merge(other_data_df, 
                                 current_google_trends_data_df,
                                 on="TIMESTAMP",
                                 how="outer").rename(columns={f"{cc}_GOOGLE_TRENDS_FULL_NAME":f"{cc}_GOOGLE_TRENDS_FULL_NAME_PER_BIC"})
    
    if os.path.exists(current_twitter_data_fp):
        current_twitter_data_fn = os.listdir(current_twitter_data_fp)[0]
        current_twitter_data_df = pd.read_csv(f"{current_twitter_data_fp}{current_twitter_data_fn}")
        
        other_data_df = pd.merge(other_data_df, 
                                 current_twitter_data_df,
                                 on="TIMESTAMP",
                                 how="outer").rename(columns={f"{cc}_TWEET_VOLUME_FULL_NAME_HASHTAG":f"{cc}_TWEET_VOLUME_FULL_NAME_HASHTAG_PER_BIC"})

    current_other_data_fp = f"All_Crypto_Data/New_Data/Unprocessed/Other_Social_Data/All_Other_Social_Data/{cc}/1_Day/"
    other_data_df['TIMESTAMP'] = pd.to_datetime(other_data_df['TIMESTAMP'])

    if not os.path.exists(current_other_data_fp):
        os.makedirs(current_other_data_fp)
    
    other_data_df.to_csv(f"{current_other_data_fp}All_Other_Social_Data_{cc}_USD_Daily_{other_data_df['TIMESTAMP'].min().strftime('%d_%m_%Y')}__{other_data_df['TIMESTAMP'].max().strftime('%d_%m_%Y')}.csv",
                         index=False)

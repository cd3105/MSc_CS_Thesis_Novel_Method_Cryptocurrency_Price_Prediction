import os
import re
import gc
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

def post_text_length_extraction(row):
    return len(row['POST_TEXT'])
    

def post_word_count_extraction(row):
    return len(re.findall(r'\b\w+\b', row['POST_TEXT']))


def comment_text_length_extraction(row):
    return len(row['COMMENT_TEXT'])
    

def comment_word_count_extraction(row):
    return len(re.findall(r'\b\w+\b', row['COMMENT_TEXT']))


def aggregation_df(df, frequency='1D'):
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values("TIMESTAMP").reset_index()

    if 'POST_TEXT' in df.columns:
        prefix = "POST"
        df[f'{prefix}_WORD_COUNT'] = df.apply(post_word_count_extraction, 
                                              axis=1)
        df[f'{prefix}_TEXT_LENGTH'] = df.apply(post_text_length_extraction, 
                                               axis=1)
    else:
        prefix = "COMMENT"
        df[f'{prefix}_WORD_COUNT'] = df.apply(comment_word_count_extraction, 
                                              axis=1)
        df[f'{prefix}_TEXT_LENGTH'] = df.apply(comment_text_length_extraction, 
                                               axis=1)

    aggregated_df_part_1 = df.groupby(pd.Grouper(key='TIMESTAMP', freq=frequency)).agg({
        f'TIMESTAMP': 'count',
        f'{prefix}_SCORE': 'sum',
        f'{prefix}_WORD_COUNT': 'sum',
        f'{prefix}_TEXT_LENGTH': 'sum',
        }).rename(columns={
        f'TIMESTAMP': f'{prefix}_COUNT',    
        f'{prefix}_SCORE': f'{prefix}_TOTAL_SCORE',
        f'{prefix}_WORD_COUNT': f'{prefix}_TOTAL_WORD_COUNT',
        f'{prefix}_TEXT_LENGTH': f'{prefix}_TOTAL_TEXT_LENGTH',
    }).reset_index()

    aggregated_df_part_2 = df.groupby(pd.Grouper(key='TIMESTAMP', freq=frequency)).agg({
        f'{prefix}_SCORE': 'mean',
        f'{prefix}_WORD_COUNT': 'mean',
        f'{prefix}_TEXT_LENGTH': 'mean',
    }).rename(columns={
        f'{prefix}_SCORE': f'{prefix}_MEAN_SCORE',
        f'{prefix}_WORD_COUNT': f'{prefix}_MEAN_WORD_COUNT',
        f'{prefix}_TEXT_LENGTH': f'{prefix}_MEAN_TEXT_LENGTH',
    }).reset_index()

    aggregated_df_part_3 = df.groupby(pd.Grouper(key='TIMESTAMP', freq=frequency)).agg({
        f'{prefix}_WORD_COUNT': 'max',
        f'{prefix}_TEXT_LENGTH': 'max',
    }).rename(columns={
        f'{prefix}_WORD_COUNT': f'{prefix}_MAX_WORD_COUNT',
        f'{prefix}_TEXT_LENGTH': f'{prefix}_MAX_TEXT_LENGTH',
    }).reset_index()

    return pd.merge(pd.merge(aggregated_df_part_1, 
                             aggregated_df_part_2, 
                             on='TIMESTAMP'), 
                             aggregated_df_part_3, 
                             on='TIMESTAMP').fillna(0)

base_file_path = "All_Crypto_Data/Crypto_Sub_Reddit_Data/Shortened_Preprocessed_Sub_Reddit_Text_Data/"

frequency_mapping_1 = {"1D": "1_Day",
                       "12H": "12_Hours",
                       "8H": "8_Hours",
                       "6H": "6_Hours",
                       "4H": "4_Hours",
                       "2H": "2_Hours",
                       '1H': "1_Hour"}

frequency_mapping_2 = {"1D": "Daily",
                       "12H": "Every_12_Hours",
                       "8H": "Every_8_Hours",
                       "6H": "Every_6_Hours",
                       "4H": "Every_4_Hours",
                       "2H": "Every_2_Hours",
                       "1H": "Hourly"}

cc_mapping = {"BTC": "Bitcoin",
              "ETH": "Ethereum",
              "LTC": "Litecoin",
              "XRP": "Ripple",
              "XMR": "Monero"}

for pm in os.listdir(base_file_path):
    print(f"Current Preprocessing Method: {pm}")

    for fk in list(frequency_mapping_1.keys()):
        print(f"\t- Current Interval: {fk}")

        cryptocurrencies = []
        reddit_data = []

        mean_post_total_score = []
        mean_post_total_score_excl_0_post_intervals = []
        mean_post_total_word_count = []
        mean_post_total_word_count_excl_0_post_intervals = []
        mean_post_total_text_length = []
        mean_post_total_text_length_excl_0_post_intervals = []
        mean_post_total_word_count = []
        mean_post_total_word_count_excl_0_post_intervals = []
        mean_post_count = []
        mean_post_count_excl_0_post_intervals = []
        mean_post_score = []
        mean_post_score_excl_0_post_intervals = []
        mean_post_word_count = []
        mean_post_word_count_excl_0_post_intervals = []
        mean_post_text_length = []
        mean_post_text_length_excl_0_post_intervals = []
        mean_max_post_word_count = []
        mean_max_post_word_count_excl_0_post_intervals = []
        mean_max_post_text_length = []
        mean_max_post_text_length_excl_0_post_intervals = []

        mean_comment_total_score = []
        mean_comment_total_score_excl_0_comment_intervals = []
        mean_comment_total_word_count = []
        mean_comment_total_word_count_excl_0_comment_intervals = []
        mean_comment_total_text_length = []
        mean_comment_total_text_length_excl_0_comment_intervals = []
        mean_comment_count = []
        mean_comment_count_excl_0_comment_intervals = []
        mean_comment_score = []
        mean_comment_score_excl_0_comment_intervals = []
        mean_comment_word_count = []
        mean_comment_word_count_excl_0_comment_intervals = []
        mean_comment_text_length = []
        mean_comment_text_length_excl_0_comment_intervals = []
        mean_max_comment_word_count = []
        mean_max_comment_word_count_excl_0_comment_intervals = []
        mean_max_comment_text_length = []
        mean_max_comment_text_length_excl_0_comment_intervals = []

        mean_all_message_total_score = []
        mean_all_message_total_score_excl_0_message_intervals = []
        mean_all_message_total_word_count = []
        mean_all_message_total_word_count_excl_0_message_intervals = []
        mean_all_message_total_text_length = []
        mean_all_message_total_text_length_excl_0_message_intervals = []
        mean_all_message_count = []
        mean_all_message_count_excl_0_message_intervals = []
        mean_all_message_score = []
        mean_all_message_score_excl_0_message_intervals = []
        mean_all_message_word_count = []
        mean_all_message_word_count_excl_0_message_intervals = []
        mean_all_message_text_length = []
        mean_all_message_text_length_excl_0_message_intervals = []
        mean_max_all_message_word_count = []
        mean_max_all_message_word_count_excl_0_message_intervals = []
        mean_max_all_message_text_length = []
        mean_max_all_message_text_length_excl_0_message_intervals = []

        max_post_total_text_length = []
        max_post_text_length = []
        max_comment_total_text_length = []
        max_comment_text_length = []
        max_all_message_total_text_length = []
        max_all_message_text_length = []

        max_post_total_word_count = []
        max_post_word_count = []
        max_comment_total_word_count = []
        max_comment_word_count = []
        max_all_message_total_word_count = []
        max_all_message_word_count = []

        total_post_count = []
        total_comment_count = []
        total_all_message_count = []
        intervals_without_post = []
        intervals_without_comment = []
        intervals_without_any_message = []


        for cc in ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']:
            print(f"\t\t- Current Crypto: {cc}")

            subreddits = list(set([sr.replace('_posts.csv', '').replace('_comments.csv', '') for sr in os.listdir(f"{base_file_path}{pm}/{cc}")]))

            for sr in subreddits:
                print(f"\t\t\t- Current Subreddit: {sr}")
                
                current_sr_posts_df = pd.read_csv(f"{base_file_path}{pm}/{cc}/{sr}_posts.csv")
                current_sr_comments_df = pd.read_csv(f"{base_file_path}{pm}/{cc}/{sr}_comments.csv")
                current_sr_aggregation_subreddit_fp = f"All_Crypto_Data/Crypto_Sub_Reddit_Data/Shortened_Preprocessed_Sub_Reddit_Text_Data_Aggregated_Stats/Unmerged/{pm}/{cc}/{frequency_mapping_1[fk]}/"
                current_sr_aggregation_subreddit_fn = f"{sr}_{frequency_mapping_2[fk]}_Aggregated_Stats.csv"

                if len(current_sr_posts_df) != 0:
                    current_sr_aggregation_posts_df = aggregation_df(current_sr_posts_df,
                                                                    frequency=fk)
                else:
                    current_sr_aggregation_posts_df = pd.DataFrame(columns=['TIMESTAMP',
                                                                            'POST_COUNT',
                                                                            'POST_TOTAL_SCORE', 
                                                                            'POST_TOTAL_WORD_COUNT', 
                                                                            'POST_TOTAL_TEXT_LENGTH', 
                                                                            'POST_MEAN_SCORE',
                                                                            'POST_MEAN_WORD_COUNT',
                                                                            'POST_MEAN_TEXT_LENGTH',
                                                                            'POST_MAX_WORD_COUNT',
                                                                            'POST_MAX_TEXT_LENGTH',])

                if len(current_sr_comments_df) != 0:
                    current_sr_aggregation_comments_df = aggregation_df(current_sr_comments_df,
                                                                        frequency=fk)
                else:
                    current_sr_aggregation_comments_df = pd.DataFrame(columns=['TIMESTAMP',
                                                                               'COMMENT_COUNT',
                                                                               'COMMENT_TOTAL_SCORE', 
                                                                               'COMMENT_TOTAL_WORD_COUNT', 
                                                                               'COMMENT_TOTAL_TEXT_LENGTH', 
                                                                               'COMMENT_MEAN_SCORE',
                                                                               'COMMENT_MEAN_WORD_COUNT',
                                                                               'COMMENT_MEAN_TEXT_LENGTH',
                                                                               'COMMENT_MAX_WORD_COUNT',
                                                                               'COMMENT_MAX_TEXT_LENGTH',])
                
                del current_sr_posts_df
                del current_sr_comments_df
                gc.collect()
                
                current_sr_aggregation_subreddit_df = pd.merge(current_sr_aggregation_posts_df,
                                                               current_sr_aggregation_comments_df,
                                                               on='TIMESTAMP',
                                                               how='outer').fillna(0)
                
                current_sr_date_range_df = pd.DataFrame({'TIMESTAMP':pd.date_range(start=current_sr_aggregation_subreddit_df['TIMESTAMP'].min(),
                                                                                   end=current_sr_aggregation_subreddit_df['TIMESTAMP'].max(),
                                                                                   freq=fk)})
                
                current_sr_aggregation_subreddit_df = pd.merge(current_sr_aggregation_subreddit_df,
                                                               current_sr_date_range_df,
                                                               on='TIMESTAMP',
                                                               how='right').fillna(0)
                
                current_sr_aggregation_subreddit_df['ALL_MESSAGES_COUNT'] = current_sr_aggregation_subreddit_df.apply(lambda x: x.POST_COUNT + x.COMMENT_COUNT, 
                                                                                                                      axis=1)
                current_sr_aggregation_subreddit_df['ALL_MESSAGES_TOTAL_SCORE'] = current_sr_aggregation_subreddit_df.apply(lambda x: x.POST_TOTAL_SCORE + x.COMMENT_TOTAL_SCORE, 
                                                                                                                            axis=1)
                current_sr_aggregation_subreddit_df['ALL_MESSAGES_TOTAL_WORD_COUNT'] = current_sr_aggregation_subreddit_df.apply(lambda x: x.POST_TOTAL_WORD_COUNT + x.COMMENT_TOTAL_WORD_COUNT, 
                                                                                                                                 axis=1)
                current_sr_aggregation_subreddit_df['ALL_MESSAGES_TOTAL_TEXT_LENGTH'] = current_sr_aggregation_subreddit_df.apply(lambda x: x.POST_TOTAL_TEXT_LENGTH + x.COMMENT_TOTAL_TEXT_LENGTH, 
                                                                                                                                  axis=1)
                current_sr_aggregation_subreddit_df['ALL_MESSAGES_MEAN_SCORE'] = current_sr_aggregation_subreddit_df.apply(lambda x: (x.POST_MEAN_SCORE + x.COMMENT_MEAN_SCORE) / 2, 
                                                                                                                           axis=1)
                current_sr_aggregation_subreddit_df['ALL_MESSAGES_MEAN_WORD_COUNT'] = current_sr_aggregation_subreddit_df.apply(lambda x: (x.POST_MEAN_WORD_COUNT + x.COMMENT_MEAN_WORD_COUNT) / 2, 
                                                                                                                                axis=1)
                current_sr_aggregation_subreddit_df['ALL_MESSAGES_MEAN_TEXT_LENGTH'] = current_sr_aggregation_subreddit_df.apply(lambda x: (x.POST_MEAN_TEXT_LENGTH + x.COMMENT_MEAN_TEXT_LENGTH) / 2, 
                                                                                                                                 axis=1)
                current_sr_aggregation_subreddit_df['ALL_MESSAGES_MAX_WORD_COUNT'] = current_sr_aggregation_subreddit_df.apply(lambda x: max(x.POST_MAX_WORD_COUNT, x.COMMENT_MAX_WORD_COUNT), 
                                                                                                                               axis=1)
                current_sr_aggregation_subreddit_df['ALL_MESSAGES_MAX_TEXT_LENGTH'] = current_sr_aggregation_subreddit_df.apply(lambda x: max(x.POST_MAX_TEXT_LENGTH, x.COMMENT_MAX_TEXT_LENGTH), 
                                                                                                                                axis=1)
                
                if not os.path.exists(current_sr_aggregation_subreddit_fp):
                    os.makedirs(current_sr_aggregation_subreddit_fp)

                current_sr_aggregation_subreddit_df.to_csv(f"{current_sr_aggregation_subreddit_fp}{current_sr_aggregation_subreddit_fn}", 
                                                           index=False)
                
                cryptocurrencies.append(cc)
                reddit_data.append('r/' + sr)

                mean_post_count.append(current_sr_aggregation_subreddit_df['POST_COUNT'].mean())
                mean_post_count_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['POST_COUNT'] != 0]['POST_COUNT'].mean())
                mean_comment_count.append(current_sr_aggregation_subreddit_df['COMMENT_COUNT'].mean())
                mean_comment_count_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['COMMENT_COUNT'] != 0]['COMMENT_COUNT'].mean())
                mean_all_message_count.append(current_sr_aggregation_subreddit_df['ALL_MESSAGES_COUNT'].mean())
                mean_all_message_count_excl_0_message_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['ALL_MESSAGES_COUNT'] != 0]['ALL_MESSAGES_COUNT'].mean())

                mean_post_total_score.append(current_sr_aggregation_subreddit_df['POST_TOTAL_SCORE'].mean())
                mean_post_total_score_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['POST_COUNT'] != 0]['POST_TOTAL_SCORE'].mean())
                mean_comment_total_score.append(current_sr_aggregation_subreddit_df['COMMENT_TOTAL_SCORE'].mean())
                mean_comment_total_score_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['COMMENT_COUNT'] != 0]['COMMENT_TOTAL_SCORE'].mean())
                mean_all_message_total_score.append(current_sr_aggregation_subreddit_df['ALL_MESSAGES_TOTAL_SCORE'].mean())
                mean_all_message_total_score_excl_0_message_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['ALL_MESSAGES_COUNT'] != 0]['ALL_MESSAGES_TOTAL_SCORE'].mean())

                mean_post_total_word_count.append(current_sr_aggregation_subreddit_df['POST_TOTAL_WORD_COUNT'].mean())
                mean_post_total_word_count_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['POST_COUNT'] != 0]['POST_TOTAL_WORD_COUNT'].mean())
                mean_comment_total_word_count.append(current_sr_aggregation_subreddit_df['COMMENT_TOTAL_WORD_COUNT'].mean())
                mean_comment_total_word_count_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['COMMENT_COUNT'] != 0]['COMMENT_TOTAL_WORD_COUNT'].mean())
                mean_all_message_total_word_count.append(current_sr_aggregation_subreddit_df['ALL_MESSAGES_TOTAL_WORD_COUNT'].mean())
                mean_all_message_total_word_count_excl_0_message_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['ALL_MESSAGES_COUNT'] != 0]['ALL_MESSAGES_TOTAL_WORD_COUNT'].mean())

                mean_post_total_text_length.append(current_sr_aggregation_subreddit_df['POST_TOTAL_TEXT_LENGTH'].mean())
                mean_post_total_text_length_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['POST_COUNT'] != 0]['POST_TOTAL_TEXT_LENGTH'].mean())
                mean_comment_total_text_length.append(current_sr_aggregation_subreddit_df['COMMENT_TOTAL_TEXT_LENGTH'].mean())
                mean_comment_total_text_length_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['COMMENT_COUNT'] != 0]['COMMENT_TOTAL_TEXT_LENGTH'].mean())
                mean_all_message_total_text_length.append(current_sr_aggregation_subreddit_df['ALL_MESSAGES_TOTAL_TEXT_LENGTH'].mean())
                mean_all_message_total_text_length_excl_0_message_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['ALL_MESSAGES_COUNT'] != 0]['ALL_MESSAGES_TOTAL_TEXT_LENGTH'].mean())

                mean_post_score.append(current_sr_aggregation_subreddit_df['POST_MEAN_SCORE'].mean())
                mean_post_score_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['POST_COUNT'] != 0]['POST_MEAN_SCORE'].mean())
                mean_comment_score.append(current_sr_aggregation_subreddit_df['COMMENT_MEAN_SCORE'].mean())
                mean_comment_score_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['COMMENT_COUNT'] != 0]['COMMENT_MEAN_SCORE'].mean())
                mean_all_message_score.append(current_sr_aggregation_subreddit_df['ALL_MESSAGES_MEAN_SCORE'].mean())
                mean_all_message_score_excl_0_message_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['ALL_MESSAGES_COUNT'] != 0]['ALL_MESSAGES_MEAN_SCORE'].mean())

                mean_post_word_count.append(current_sr_aggregation_subreddit_df['POST_MEAN_WORD_COUNT'].mean())
                mean_post_word_count_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['POST_COUNT'] != 0]['POST_MEAN_WORD_COUNT'].mean())
                mean_comment_word_count.append(current_sr_aggregation_subreddit_df['COMMENT_MEAN_WORD_COUNT'].mean())
                mean_comment_word_count_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['COMMENT_COUNT'] != 0]['COMMENT_MEAN_WORD_COUNT'].mean())
                mean_all_message_word_count.append(current_sr_aggregation_subreddit_df['ALL_MESSAGES_MEAN_WORD_COUNT'].mean())
                mean_all_message_word_count_excl_0_message_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['ALL_MESSAGES_COUNT'] != 0]['ALL_MESSAGES_MEAN_WORD_COUNT'].mean())

                mean_post_text_length.append(current_sr_aggregation_subreddit_df['POST_MEAN_TEXT_LENGTH'].mean())
                mean_post_text_length_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['POST_COUNT'] != 0]['POST_MEAN_TEXT_LENGTH'].mean())
                mean_comment_text_length.append(current_sr_aggregation_subreddit_df['COMMENT_MEAN_TEXT_LENGTH'].mean())
                mean_comment_text_length_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['COMMENT_COUNT'] != 0]['COMMENT_MEAN_TEXT_LENGTH'].mean())
                mean_all_message_text_length.append(current_sr_aggregation_subreddit_df['ALL_MESSAGES_MEAN_TEXT_LENGTH'].mean())
                mean_all_message_text_length_excl_0_message_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['ALL_MESSAGES_COUNT'] != 0]['ALL_MESSAGES_MEAN_TEXT_LENGTH'].mean())

                mean_max_post_word_count.append(current_sr_aggregation_subreddit_df['POST_MAX_WORD_COUNT'].mean())
                mean_max_post_word_count_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['POST_COUNT'] != 0]['POST_MAX_WORD_COUNT'].mean())
                mean_max_comment_word_count.append(current_sr_aggregation_subreddit_df['COMMENT_MAX_WORD_COUNT'].mean())
                mean_max_comment_word_count_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['COMMENT_COUNT'] != 0]['COMMENT_MAX_WORD_COUNT'].mean())
                mean_max_all_message_word_count.append(current_sr_aggregation_subreddit_df['ALL_MESSAGES_MAX_WORD_COUNT'].mean())
                mean_max_all_message_word_count_excl_0_message_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['ALL_MESSAGES_COUNT'] != 0]['ALL_MESSAGES_MAX_WORD_COUNT'].mean())

                mean_max_post_text_length.append(current_sr_aggregation_subreddit_df['POST_MAX_TEXT_LENGTH'].mean())
                mean_max_post_text_length_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['POST_COUNT'] != 0]['POST_MAX_TEXT_LENGTH'].mean())
                mean_max_comment_text_length.append(current_sr_aggregation_subreddit_df['COMMENT_MAX_TEXT_LENGTH'].mean())
                mean_max_comment_text_length_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['COMMENT_COUNT'] != 0]['COMMENT_MAX_TEXT_LENGTH'].mean())
                mean_max_all_message_text_length.append(current_sr_aggregation_subreddit_df['ALL_MESSAGES_MAX_TEXT_LENGTH'].mean())
                mean_max_all_message_text_length_excl_0_message_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['ALL_MESSAGES_COUNT'] != 0]['ALL_MESSAGES_MAX_TEXT_LENGTH'].mean())

                max_post_total_word_count.append(current_sr_aggregation_subreddit_df['POST_TOTAL_WORD_COUNT'].max())
                max_post_word_count.append(current_sr_aggregation_subreddit_df['POST_MAX_WORD_COUNT'].max())
                max_comment_total_word_count.append(current_sr_aggregation_subreddit_df['COMMENT_TOTAL_WORD_COUNT'].max())
                max_comment_word_count.append(current_sr_aggregation_subreddit_df['COMMENT_MAX_WORD_COUNT'].max())
                max_all_message_total_word_count.append(current_sr_aggregation_subreddit_df['ALL_MESSAGES_TOTAL_WORD_COUNT'].max())
                max_all_message_word_count.append(current_sr_aggregation_subreddit_df['ALL_MESSAGES_MAX_WORD_COUNT'].max())

                max_post_total_text_length.append(current_sr_aggregation_subreddit_df['POST_TOTAL_TEXT_LENGTH'].max())
                max_post_text_length.append(current_sr_aggregation_subreddit_df['POST_MAX_TEXT_LENGTH'].max())
                max_comment_total_text_length.append(current_sr_aggregation_subreddit_df['COMMENT_TOTAL_TEXT_LENGTH'].max())
                max_comment_text_length.append(current_sr_aggregation_subreddit_df['COMMENT_MAX_TEXT_LENGTH'].max())
                max_all_message_total_text_length.append(current_sr_aggregation_subreddit_df['ALL_MESSAGES_TOTAL_TEXT_LENGTH'].max())
                max_all_message_text_length.append(current_sr_aggregation_subreddit_df['ALL_MESSAGES_MAX_TEXT_LENGTH'].max())

                total_post_count.append(current_sr_aggregation_subreddit_df['POST_COUNT'].sum())
                intervals_without_post.append(len(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['POST_COUNT'] == 0]))
                total_comment_count.append(current_sr_aggregation_subreddit_df['COMMENT_COUNT'].sum())
                intervals_without_comment.append(len(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['COMMENT_COUNT'] == 0]))
                total_all_message_count.append(current_sr_aggregation_subreddit_df['ALL_MESSAGES_COUNT'].sum())
                intervals_without_any_message.append(len(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['ALL_MESSAGES_COUNT'] == 0]))

                del current_sr_aggregation_subreddit_df
                gc.collect()
        
        total_aggregation_fp = f"All_Crypto_Data/Crypto_Sub_Reddit_Data/Shortened_Preprocessed_Sub_Reddit_Text_Data_Aggregated_Stats/Merged/{pm}/All_CCs/{frequency_mapping_1[fk]}/"
        total_aggregation_fn = f"All_Subreddits_Shortened_Preprocessed_Sub_Reddit_Text_Data_{frequency_mapping_2[fk]}_Aggregated_Stats.csv"

        if not os.path.exists(total_aggregation_fp):
            os.makedirs(total_aggregation_fp)

        pd.DataFrame({'CRYPTOCURRENCY': cryptocurrencies,
                      'REDDIT_DATA': reddit_data,

                      'MEAN_POST_COUNT': mean_post_count,
                      'MEAN_POST_COUNT_EXCL_0_POST_INTERVALS': mean_post_count_excl_0_post_intervals,

                      'MEAN_COMMENT_COUNT': mean_comment_count,
                      'MEAN_COMMENT_COUNT_EXCL_0_COMMENT_INTERVALS': mean_comment_count_excl_0_comment_intervals,

                      'MEAN_POST_SCORE': mean_post_score,
                      'MEAN_POST_SCORE_EXCL_0_POST_INTERVALS': mean_post_score_excl_0_post_intervals,

                      'MEAN_COMMENT_SCORE': mean_comment_score,
                      'MEAN_COMMENT_SCORE_EXCL_0_COMMENT_INTERVALS': mean_comment_score_excl_0_comment_intervals,

                      'MEAN_TOTAL_POST_SCORE': mean_post_total_score,
                      'MEAN_TOTAL_POST_SCORE_EXCL_0_POST_INTERVALS': mean_post_total_score_excl_0_post_intervals,

                      'MEAN_TOTAL_COMMENT_SCORE': mean_comment_total_score,
                      'MEAN_TOTAL_COMMENT_SCORE_EXCL_0_COMMENT_INTERVALS': mean_comment_total_score_excl_0_comment_intervals,

                      'MEAN_POST_WORD_COUNT': mean_post_word_count,
                      'MEAN_POST_WORD_COUNT_EXCL_0_POST_INTERVALS': mean_max_post_word_count_excl_0_post_intervals,
                      'MEAN_MAX_POST_WORD_COUNT': mean_post_word_count,
                      'MEAN_MAX_POST_WORD_COUNT_EXCL_0_POST_INTERVALS': mean_max_post_word_count_excl_0_post_intervals,
                      'MAX_POST_WORD_COUNT': max_post_word_count,

                      'MEAN_COMMENT_WORD_COUNT': mean_comment_word_count,
                      'MEAN_COMMENT_WORD_COUNT_EXCL_0_COMMENT_INTERVALS': mean_comment_word_count_excl_0_comment_intervals,
                      'MEAN_MAX_COMMENT_WORD_COUNT': mean_max_comment_word_count,
                      'MEAN_MAX_COMMENT_WORD_COUNT_EXCL_0_COMMENT_INTERVALS': mean_max_comment_word_count_excl_0_comment_intervals,
                      'MAX_COMMENT_WORD_COUNT': max_comment_word_count,

                      'MEAN_POST_TEXT_LENGTH': mean_post_text_length,
                      'MEAN_POST_TEXT_LENGTH_EXCL_0_POST_INTERVALS': mean_max_post_text_length_excl_0_post_intervals,
                      'MEAN_MAX_POST_TEXT_LENGTH': mean_post_text_length,
                      'MEAN_MAX_POST_TEXT_LENGTH_EXCL_0_POST_INTERVALS': mean_max_post_text_length_excl_0_post_intervals,
                      'MAX_POST_TEXT_LENGTH': max_post_text_length,

                      'MEAN_COMMENT_TEXT_LENGTH': mean_comment_text_length,
                      'MEAN_COMMENT_TEXT_LENGTH_EXCL_0_COMMENT_INTERVALS': mean_comment_text_length_excl_0_comment_intervals,
                      'MEAN_MAX_COMMENT_TEXT_LENGTH': mean_max_comment_text_length,
                      'MEAN_MAX_COMMENT_TEXT_LENGTH_EXCL_0_COMMENT_INTERVALS': mean_max_comment_text_length_excl_0_comment_intervals,
                      'MAX_COMMENT_TEXT_LENGTH': max_comment_text_length,

                      'MEAN_TOTAL_POST_WORD_COUNT': mean_post_total_word_count,
                      'MEAN_TOTAL_POST_WORD_COUNT_EXCL_0_POST_INTERVALS': mean_post_total_word_count_excl_0_post_intervals,
                      'MAX_TOTAL_POST_WORD_COUNT': max_post_total_word_count,

                      'MEAN_TOTAL_COMMENT_WORD_COUNT': mean_comment_total_word_count,
                      'MEAN_TOTAL_COMMENT_WORD_COUNT_EXCL_0_COMMENT_INTERVALS': mean_comment_total_word_count_excl_0_comment_intervals,
                      'MAX_TOTAL_COMMENT_WORD_COUNT': max_comment_total_word_count,

                      'MEAN_TOTAL_POST_TEXT_LENGTH': mean_post_total_text_length,
                      'MEAN_TOTAL_POST_TEXT_LENGTH_EXCL_0_POST_INTERVALS': mean_post_total_text_length_excl_0_post_intervals,
                      'MAX_TOTAL_POST_TEXT_LENGTH': max_post_total_text_length,

                      'MEAN_TOTAL_COMMENT_TEXT_LENGTH': mean_comment_total_text_length,
                      'MEAN_TOTAL_COMMENT_TEXT_LENGTH_EXCL_0_COMMENT_INTERVALS': mean_comment_total_text_length_excl_0_comment_intervals,
                      'MAX_TOTAL_COMMENT_TEXT_LENGTH': max_comment_total_text_length,

                      'TOTAL_POST_COUNT': total_post_count,
                      'TOTAL_COMMENT_COUNT': total_comment_count,
                      'TOTAL_INTERVALS_WO_POST': intervals_without_post,
                      'TOTAL_INTERVALS_WO_COMMENT': intervals_without_comment,}).to_csv(f"{total_aggregation_fp}{total_aggregation_fn}", 
                                                                                        index=False)


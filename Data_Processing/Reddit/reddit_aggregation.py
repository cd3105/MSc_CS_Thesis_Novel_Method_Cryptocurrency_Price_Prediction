import os
import re
import gc
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

def submission_text_length_extraction(row):
    if ((row['title'] != '[removed]') and (row['text'] != '[removed]')) and ((row['title'] != '[deleted]') and (row['text'] != '[deleted]')) and ((type(row['title']) == str) and (type(row['text']) == str)):
        return (len(row['title']) + len(row['text']))
    elif (row['title'] != '[removed]') and (row['title'] != '[deleted]') and (type(row['title']) == str):
        return len(row['title'])
    elif (row['text'] != '[removed]') and (row['text'] != '[deleted]') and (type(row['text']) == str):
        return len(row['text'])
    else:
        return 0


def comment_text_length_extraction(row):
    if (row['body'] != '[removed]') and (row['body'] != '[deleted]') and (type(row['body']) == str):
        return len(row['body'])
    else:
        return 0


def aggregation_df(df, frequency='1D'):
    df['created'] = pd.to_datetime(df['created'], format="%H:%M %d-%m-%Y")
    df = df.sort_values("created").reset_index()

    if 'title' in df.columns:
        prefix = "Post"
        df['Text_Length'] = df.apply(submission_text_length_extraction, axis=1)
    else:
        prefix = "Comment"
        df['Text_Length'] = df.apply(comment_text_length_extraction, axis=1)

    aggregated_df_part_1 = df.groupby(pd.Grouper(key='created', freq=frequency)).agg({
        'score': 'sum',
        'Text_Length': 'sum',
        'link': 'count',
        }).rename(columns={
        'score': prefix + '_Total_Score',
        'Text_Length': prefix + '_Total_Text_Length',
        'link': prefix + '_Count',
    }).reset_index()

    aggregated_df_part_2 = df.groupby(pd.Grouper(key='created', freq=frequency)).agg({
        'score': 'mean',
        'Text_Length': 'mean'
    }).rename(columns={
        'score': prefix + '_Average_Score',
        'Text_Length': prefix + '_Average_Text_Length'
    }).reset_index()

    aggregated_df_part_3 = df.groupby(pd.Grouper(key='created', freq=frequency)).agg({
        'Text_Length': 'max'
    }).rename(columns={
        'Text_Length': prefix + '_Max_Text_Length'
    }).reset_index()

    return pd.merge(pd.merge(aggregated_df_part_1, aggregated_df_part_2, on='created'), aggregated_df_part_3, on='created').fillna(0)

base_file_path = "All_Crypto_Data/Crypto_Sub_Reddit_Data/"

frequency_mapping = {"1D": "1_Day",
                     "12H": "12_Hours",
                     "8H": "8_Hours",
                     "6H": "6_Hours",
                     "4H": "4_Hours",
                     "2H": "2_Hours",
                     '1H': "1_Hour",
                     "30Min": "30_Minutes",
                     "15Min": "15_Minutes",
                     "1Min": "1_Minutes"}

cc_mapping = {"BTC": "Bitcoin",
              "ETH": "Ethereum",
              "LTC": "Litecoin",
              "XRP": "Ripple",
              "XMR": "Monero"}

for fk in list(frequency_mapping.keys()):
    cryptocurrencies = []
    reddit_data = []
    mean_post_total_score = []
    mean_post_total_score_excl_0_post_intervals = []
    mean_post_total_text_length = []
    mean_post_total_text_length_excl_0_post_intervals = []
    mean_post_count = []
    mean_post_count_excl_0_post_intervals = []
    mean_post_score = []
    mean_post_score_excl_0_post_intervals = []
    mean_post_text_length = []
    mean_post_text_length_excl_0_post_intervals = []
    mean_max_post_text_length = []
    mean_max_post_text_length_excl_0_post_intervals = []

    mean_comment_total_score = []
    mean_comment_total_score_excl_0_comment_intervals = []
    mean_comment_total_text_length = []
    mean_comment_total_text_length_excl_0_comment_intervals = []
    mean_comment_count = []
    mean_comment_count_excl_0_comment_intervals = []
    mean_comment_score = []
    mean_comment_score_excl_0_comment_intervals = []
    mean_comment_text_length = []
    mean_comment_text_length_excl_0_comment_intervals = []
    mean_max_comment_text_length = []
    mean_max_comment_text_length_excl_0_comment_intervals = []

    max_post_total_text_length = []
    max_post_text_length = []
    max_comment_total_text_length = []
    max_comment_text_length = []

    total_post_count = []
    total_comment_count = []
    intervals_without_post = []
    intervals_without_comment = []

    for cc in ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']:
        subreddits = sorted(list(set([re.sub(r'_(submissions|comments)\.csv$', '', sr)
                                      for sr in os.listdir(base_file_path + cc + "/Extracted_Reddit_Data/")])))
        crypto_subreddits = sorted(list(set([re.sub(r'_(submissions|comments)\.csv$', '', csr)
                                             for csr in os.listdir(base_file_path + "Crypto" + "/Extracted_Reddit_Data/")])))

        for sr in subreddits:
            print(f"Current Crypto: {cc} & Current Subreddit: {sr} at Interval {fk}")

            current_sr_submissions_df = pd.read_csv(base_file_path + cc + "/Extracted_Reddit_Data/" + sr + "_submissions.csv")
            current_sr_comments_df = pd.read_csv(base_file_path + cc + "/Extracted_Reddit_Data/" + sr + "_comments.csv")
            current_sr_aggregation_subreddit_fp = f"{base_file_path}{cc}/Extracted_Reddit_Data_Aggregated_Stats/{frequency_mapping[fk]}/{sr}_{frequency_mapping[fk]}_Aggregated_Stats.csv"

            current_sr_aggregation_submissions_df = aggregation_df(current_sr_submissions_df,
                                                                   frequency=fk)
            current_sr_aggregation_comments_df = aggregation_df(current_sr_comments_df,
                                                                frequency=fk)

            del current_sr_submissions_df
            del current_sr_comments_df
            gc.collect()

            current_sr_aggregation_subreddit_df = pd.merge(current_sr_aggregation_submissions_df,
                                                           current_sr_aggregation_comments_df,
                                                           on='created',
                                                           how='outer').fillna(0)

            current_sr_aggregation_subreddit_df.to_csv(current_sr_aggregation_subreddit_fp)

            cryptocurrencies.append(cc)
            reddit_data.append('r/' + sr)
            mean_post_total_score.append(current_sr_aggregation_subreddit_df['Post_Total_Score'].mean())
            mean_post_total_score_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['Post_Count'] != 0]['Post_Total_Score'].mean())

            mean_post_total_text_length.append(current_sr_aggregation_subreddit_df['Post_Total_Text_Length'].mean())
            mean_post_total_text_length_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['Post_Count'] != 0]['Post_Total_Score'].mean())

            mean_post_count.append(current_sr_aggregation_subreddit_df['Post_Count'].mean())
            mean_post_count_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['Post_Count'] != 0]['Post_Count'].mean())

            mean_post_score.append(current_sr_aggregation_subreddit_df['Post_Average_Score'].mean())
            mean_post_score_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['Post_Count'] != 0]['Post_Average_Score'].mean())

            mean_post_text_length.append(current_sr_aggregation_subreddit_df['Post_Average_Text_Length'].mean())
            mean_post_text_length_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['Post_Count'] != 0]['Post_Average_Text_Length'].mean())

            mean_max_post_text_length.append(current_sr_aggregation_subreddit_df['Post_Max_Text_Length'].mean())
            mean_max_post_text_length_excl_0_post_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['Post_Count'] != 0]['Post_Max_Text_Length'].mean())

            mean_comment_total_score.append(current_sr_aggregation_subreddit_df['Comment_Total_Score'].mean())
            mean_comment_total_score_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['Comment_Count'] != 0]['Comment_Total_Score'].mean())

            mean_comment_total_text_length.append(current_sr_aggregation_subreddit_df['Comment_Total_Text_Length'].mean())
            mean_comment_total_text_length_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['Comment_Count'] != 0]['Comment_Total_Text_Length'].mean())

            mean_comment_count.append(current_sr_aggregation_subreddit_df['Comment_Count'].mean())
            mean_comment_count_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['Comment_Count'] != 0]['Comment_Count'].mean())

            mean_comment_score.append(current_sr_aggregation_subreddit_df['Comment_Average_Score'].mean())
            mean_comment_score_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['Comment_Count'] != 0]['Comment_Average_Score'].mean())

            mean_comment_text_length.append(current_sr_aggregation_subreddit_df['Comment_Average_Text_Length'].mean())
            mean_comment_text_length_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['Comment_Count'] != 0]['Comment_Average_Text_Length'].mean())

            mean_max_comment_text_length.append(current_sr_aggregation_subreddit_df['Comment_Max_Text_Length'].mean())
            mean_max_comment_text_length_excl_0_comment_intervals.append(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['Comment_Count'] != 0]['Comment_Max_Text_Length'].mean())

            max_post_total_text_length.append(current_sr_aggregation_subreddit_df['Post_Total_Text_Length'].max())
            max_post_text_length.append(current_sr_aggregation_subreddit_df['Post_Max_Text_Length'].max())
            max_comment_total_text_length.append(current_sr_aggregation_subreddit_df['Comment_Total_Text_Length'].max())
            max_comment_text_length.append(current_sr_aggregation_subreddit_df['Comment_Max_Text_Length'].max())

            total_post_count.append(current_sr_aggregation_subreddit_df['Post_Count'].sum())
            total_comment_count.append(current_sr_aggregation_subreddit_df['Comment_Count'].sum())
            intervals_without_post.append(len(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['Post_Count'] == 0]))
            intervals_without_comment.append(len(current_sr_aggregation_subreddit_df[current_sr_aggregation_subreddit_df['Comment_Count'] == 0]))

            del current_sr_aggregation_subreddit_df
            gc.collect()


        for csr in crypto_subreddits:
            print(f"Current Crypto: {cc} & Current Crypto Subreddit: {csr} at Interval {fk}")

            current_csr_submissions_df = pd.read_csv(base_file_path + "Crypto/Extracted_Reddit_Data/" + csr + "_submissions.csv")
            current_csr_comments_df = pd.read_csv(base_file_path + "Crypto/Extracted_Reddit_Data/" + csr + "_comments.csv")
            current_csr_aggregation_subreddit_fp = f"{base_file_path}{cc}/Extracted_Reddit_Data_Aggregated_Stats/{frequency_mapping[fk]}/{cc_mapping[cc]}_in_{csr}_{frequency_mapping[fk]}_Aggregated_Stats.csv"

            current_csr_submissions_df['Post_ID'] = current_csr_submissions_df['link'].apply(lambda x: f'/{x.split("/comments/")[1].split("/")[0]}/')
            current_csr_comments_df['Post_ID'] = current_csr_comments_df['link'].apply(lambda x: f'/{x.split("/comments/")[1].split("/")[0]}/')

            relevant_current_csr_submissions_df = current_csr_submissions_df[current_csr_submissions_df['text'].str.contains(fr'\b({cc}|{re.escape("$" + cc)}|{cc_mapping[cc]}|{re.escape("$" + cc_mapping[cc])})\b', case=False, na=False) |
                                                                             current_csr_submissions_df['title'].str.contains(fr'\b({cc}|{re.escape("$" + cc)}|{cc_mapping[cc]}|{re.escape("$" + cc_mapping[cc])})\b', case=False, na=False)].reset_index(drop=True)
            relevant_current_csr_comments_df = current_csr_comments_df[current_csr_comments_df['body'].str.contains(fr'\b({cc}|{re.escape("$" + cc)}|{cc_mapping[cc]}|{re.escape("$" + cc_mapping[cc])})\b', case=False, na=False) |
                                                                       current_csr_comments_df['Post_ID'].isin(list(relevant_current_csr_submissions_df['Post_ID']))].reset_index(drop=True)

            current_csr_aggregation_submissions_df = aggregation_df(relevant_current_csr_submissions_df,
                                                                    frequency=fk)
            current_csr_aggregation_comments_df = aggregation_df(relevant_current_csr_comments_df,
                                                                    frequency=fk)

            del relevant_current_csr_submissions_df
            del relevant_current_csr_comments_df
            gc.collect()

            current_csr_aggregation_subreddit_df = pd.merge(current_csr_aggregation_submissions_df,
                                                            current_csr_aggregation_comments_df,
                                                            on='created',
                                                            how='outer').fillna(0)

            current_csr_aggregation_subreddit_df.to_csv(current_csr_aggregation_subreddit_fp)

            cryptocurrencies.append(cc)
            reddit_data.append(f'{cc_mapping[cc]} in r/{csr}')
            mean_post_total_score.append(current_csr_aggregation_subreddit_df['Post_Total_Score'].mean())
            mean_post_total_score_excl_0_post_intervals.append(current_csr_aggregation_subreddit_df[current_csr_aggregation_subreddit_df['Post_Count'] != 0]['Post_Total_Score'].mean())

            mean_post_total_text_length.append(current_csr_aggregation_subreddit_df['Post_Total_Text_Length'].mean())
            mean_post_total_text_length_excl_0_post_intervals.append(current_csr_aggregation_subreddit_df[current_csr_aggregation_subreddit_df['Post_Count'] != 0]['Post_Total_Score'].mean())

            mean_post_count.append(current_csr_aggregation_subreddit_df['Post_Count'].mean())
            mean_post_count_excl_0_post_intervals.append(current_csr_aggregation_subreddit_df[current_csr_aggregation_subreddit_df['Post_Count'] != 0]['Post_Count'].mean())

            mean_post_score.append(current_csr_aggregation_subreddit_df['Post_Average_Score'].mean())
            mean_post_score_excl_0_post_intervals.append(current_csr_aggregation_subreddit_df[current_csr_aggregation_subreddit_df['Post_Count'] != 0]['Post_Average_Score'].mean())

            mean_post_text_length.append(current_csr_aggregation_subreddit_df['Post_Average_Text_Length'].mean())
            mean_post_text_length_excl_0_post_intervals.append(current_csr_aggregation_subreddit_df[current_csr_aggregation_subreddit_df['Post_Count'] != 0]['Post_Average_Text_Length'].mean())

            mean_max_post_text_length.append(current_csr_aggregation_subreddit_df['Post_Max_Text_Length'].mean())
            mean_max_post_text_length_excl_0_post_intervals.append(current_csr_aggregation_subreddit_df[current_csr_aggregation_subreddit_df['Post_Count'] != 0]['Post_Max_Text_Length'].mean())

            mean_comment_total_score.append(current_csr_aggregation_subreddit_df['Comment_Total_Score'].mean())
            mean_comment_total_score_excl_0_comment_intervals.append(current_csr_aggregation_subreddit_df[current_csr_aggregation_subreddit_df['Comment_Count'] != 0]['Comment_Total_Score'].mean())

            mean_comment_total_text_length.append(current_csr_aggregation_subreddit_df['Comment_Total_Text_Length'].mean())
            mean_comment_total_text_length_excl_0_comment_intervals.append(current_csr_aggregation_subreddit_df[current_csr_aggregation_subreddit_df['Comment_Count'] != 0]['Comment_Total_Text_Length'].mean())

            mean_comment_count.append(current_csr_aggregation_subreddit_df['Comment_Count'].mean())
            mean_comment_count_excl_0_comment_intervals.append(current_csr_aggregation_subreddit_df[current_csr_aggregation_subreddit_df['Comment_Count'] != 0]['Comment_Count'].mean())

            mean_comment_score.append(current_csr_aggregation_subreddit_df['Comment_Average_Score'].mean())
            mean_comment_score_excl_0_comment_intervals.append(current_csr_aggregation_subreddit_df[current_csr_aggregation_subreddit_df['Comment_Count'] != 0]['Comment_Average_Score'].mean())

            mean_comment_text_length.append(current_csr_aggregation_subreddit_df['Comment_Average_Text_Length'].mean())
            mean_comment_text_length_excl_0_comment_intervals.append(current_csr_aggregation_subreddit_df[current_csr_aggregation_subreddit_df['Comment_Count'] != 0]['Comment_Average_Text_Length'].mean())

            mean_max_comment_text_length.append(current_csr_aggregation_subreddit_df['Comment_Max_Text_Length'].mean())
            mean_max_comment_text_length_excl_0_comment_intervals.append( current_csr_aggregation_subreddit_df[current_csr_aggregation_subreddit_df['Comment_Count'] != 0]['Comment_Max_Text_Length'].mean())

            max_post_total_text_length.append(current_csr_aggregation_subreddit_df['Post_Total_Text_Length'].max())
            max_post_text_length.append(current_csr_aggregation_subreddit_df['Post_Average_Text_Length'].max())
            max_comment_total_text_length.append(current_csr_aggregation_subreddit_df['Comment_Total_Text_Length'].max())
            max_comment_text_length.append(current_csr_aggregation_subreddit_df['Comment_Average_Text_Length'].max())

            total_post_count.append(current_csr_aggregation_subreddit_df['Post_Count'].sum())
            total_comment_count .append(current_csr_aggregation_subreddit_df['Comment_Count'].sum())
            intervals_without_post.append(len(current_csr_aggregation_subreddit_df[current_csr_aggregation_subreddit_df['Post_Count'] == 0]))
            intervals_without_comment.append(len(current_csr_aggregation_subreddit_df[current_csr_aggregation_subreddit_df['Comment_Count'] == 0]))

            del current_csr_aggregation_subreddit_df
            gc.collect()


    pd.DataFrame({'Cryptocurrency':cryptocurrencies,
                  'Reddit_Data': reddit_data,

                  'Mean_Post_Count': mean_post_count,
                  'Mean_Post_Count_excl_0_Post_Intervals': mean_post_count_excl_0_post_intervals,

                  'Mean_Post_Score': mean_post_score,
                  'Mean_Post_Score_excl_0_Post_Intervals': mean_post_score_excl_0_post_intervals,

                  'Mean_Total_Post_Score': mean_post_total_score,
                  'Mean_Total_Post_Score_excl_0_Post_Intervals': mean_post_total_score_excl_0_post_intervals,

                  'Mean_Post_Text_Length': mean_post_text_length,
                  'Mean_Post_Text_Length_excl_0_Post_Intervals': mean_max_post_text_length_excl_0_post_intervals,
                  'Mean_Max_Post_Text_Length': mean_post_text_length,
                  'Mean_Max_Post_Text_Length_excl_0_Post_Intervals': mean_max_post_text_length_excl_0_post_intervals,
                  'Max_Post_Text_Length': max_post_text_length,

                  'Mean_Total_Post_Text_Length': mean_post_total_text_length,
                  'Mean_Total_Post_Text_Length_excl_0_Post_Intervals': mean_post_total_text_length_excl_0_post_intervals,
                  'Max_Total_Post_Text_Length': max_post_total_text_length,

                  'Mean_Comment_Count': mean_comment_count,
                  'Mean_Comment_Count_excl_0_Comment_Intervals': mean_comment_count_excl_0_comment_intervals,

                  'Mean_Comment_Score': mean_comment_score,
                  'Mean_Comment_Score_excl_0_Comment_Intervals': mean_comment_score_excl_0_comment_intervals,

                  'Mean_Total_Comment_Score': mean_comment_total_score,
                  'Mean_Total_Comment_Score_excl_0_Comment_Intervals': mean_comment_total_score_excl_0_comment_intervals,

                  'Mean_Comment_Text_Length': mean_comment_text_length,
                  'Mean_Comment_Text_Length_excl_0_Comment_Intervals': mean_comment_text_length_excl_0_comment_intervals,
                  'Mean_Max_Comment_Text_Length': mean_max_comment_text_length,
                  'Mean_Max_Comment_Text_Length_excl_0_Comment_Intervals': mean_max_comment_text_length_excl_0_comment_intervals,
                  'Max_Comment_Text_Length': max_comment_text_length,

                  'Mean_Total_Comment_Text_Length': mean_comment_total_text_length,
                  'Mean_Total_Comment_Text_Length_excl_0_Comment_Intervals': mean_comment_total_text_length_excl_0_comment_intervals,
                  'Max_Total_Comment_Text_Length': max_comment_total_text_length,

                  'Total_Post_Count': total_post_count,
                  'Total_Comment_Count': total_comment_count,
                  'Total_Intervals_wo_Post': intervals_without_post,
                  'Total_Intervals_wo_Comment': intervals_without_comment,}).to_csv(f"All_Crypto_Data/Crypto_Sub_Reddit_Data/Selected_CCs_Unpreprocessed_Reddit_Data_Aggregated_Stats/Unpreprocessed_Reddit_Data_{frequency_mapping[fk]}_Aggregated_Stats.csv")

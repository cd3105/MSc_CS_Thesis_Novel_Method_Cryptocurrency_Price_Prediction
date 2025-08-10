import pandas as pd
import os


def merge_means(mean_x, count_x, mean_y, count_y):
    if count_x == 0:
        return mean_y
    elif count_y == 0:
        return mean_x
    else:
        return sum([mean_x * count_x, mean_y * count_y]) / sum([count_x, count_y])


def merge_columns(merged_df, columns_to_merge):
    for c in columns_to_merge:
        if 'TOTAL' in c:
            merged_df[c] = merged_df.apply(lambda x: sum([x[f'{c}_x'], x[f'{c}_y']]), axis=1)
        elif 'MAX' in c:
            merged_df[c] = merged_df.apply(lambda x: max(x[f'{c}_x'], x[f'{c}_y']), axis=1)
        elif 'MEAN' in c:
            if 'POST' in c:
                prefix = 'POST'
            elif 'COMMENT' in c:
                prefix = 'COMMENT'
            else:
                prefix = 'ALL_MESSAGES'

            merged_df[c] = merged_df.apply(lambda x: merge_means(x[f'{c}_x'], x[f'{prefix}_COUNT_x'], x[f'{c}_y'], x[f'{prefix}_COUNT_y']), axis=1)
        else:
            merged_df[c] = merged_df.apply(lambda x: sum([x[f'{c}_x'], x[f'{c}_y']]), axis=1)
        
    merged_df = merged_df.drop([c for c in merged_df.columns if c.endswith('_x') or c.endswith('_y')], axis=1)

    return merged_df


base_path = "All_Crypto_Data/Crypto_Sub_Reddit_Data/Shortened_Preprocessed_Sub_Reddit_Text_Data_Aggregated_Stats/Unmerged/"
frequencies = {"1_Day": "Daily",
               "12_Hours": "Every_12_Hours",
               "8_Hours": "Every_8_Hours",
               "6_Hours": "Every_6_Hours",
               "4_Hours": "Every_4_Hours",
               "2_Hours": "Every_2_Hours",
               "1_Hour": "Hourly"}

for pm in os.listdir(base_path):
    print(f"Current Preprocessing Method: {pm}")

    for cc in os.listdir(f'{base_path}/{pm}/'):
        print(f"\t- Current Crypto: {cc}")

        for f in os.listdir(f"{base_path}{pm}/{cc}/"):
            print(f"\t\t- Current Freq: {f}")

            current_cc_path = f"{base_path}{pm}/{cc}/{f}/"
            all_sr_csvs = [csv for csv in os.listdir(current_cc_path) if not csv.startswith(f'{cc}_In')]
            all_csr_csvs = [csv for csv in os.listdir(current_cc_path) if csv.startswith(f'{cc}_In')]
            current_all_merged_df = pd.DataFrame()
            current_all_sr_merged_df = pd.DataFrame()
            current_all_csr_merged_df = pd.DataFrame()
            columns_to_merge = pd.read_csv(f'{current_cc_path}{all_sr_csvs[0]}').columns[1:]
            current_merged_fp = f"All_Crypto_Data/Crypto_Sub_Reddit_Data/Shortened_Preprocessed_Sub_Reddit_Text_Data_Aggregated_Stats/Merged/{pm}/{cc}/{f}/"

            for i, sr_csv in enumerate(all_sr_csvs):
                current_sr_df = pd.read_csv(f'{current_cc_path}{sr_csv}')

                if i==0:
                    current_all_sr_merged_df = current_sr_df
                    current_all_merged_df = current_sr_df
                else:
                    current_all_sr_merged_df = pd.merge(current_all_sr_merged_df, current_sr_df, how='outer', on='TIMESTAMP').fillna(0)
                    current_all_sr_merged_df = merge_columns(current_all_sr_merged_df, columns_to_merge)

                    current_all_merged_df = pd.merge(current_all_merged_df, current_sr_df, how='outer', on='TIMESTAMP').fillna(0)
                    current_all_merged_df = merge_columns(current_all_merged_df, columns_to_merge)

            
            for i, csr_csv in enumerate(all_csr_csvs):
                current_csr_df = pd.read_csv(f'{current_cc_path}{csr_csv}')

                if i==0:
                    current_all_csr_merged_df = current_csr_df
                else:
                    current_all_csr_merged_df = pd.merge(current_all_csr_merged_df, current_csr_df, how='outer', on='TIMESTAMP').fillna(0)
                    current_all_csr_merged_df = merge_columns(current_all_csr_merged_df, columns_to_merge)

                current_all_merged_df = pd.merge(current_all_merged_df, current_csr_df, how='outer', on='TIMESTAMP').fillna(0)
                current_all_merged_df = merge_columns(current_all_merged_df, columns_to_merge)
            
            if not os.path.exists(current_merged_fp):
                os.makedirs(current_merged_fp)
            
            current_all_sr_merged_df.to_csv(f"{current_merged_fp}All_{cc}_Subreddits_{frequencies[f]}_Aggregated_Stats.csv")
            current_all_csr_merged_df.to_csv(f"{current_merged_fp}All_{cc}_In_Crypto_Subreddits_{frequencies[f]}_Aggregated_Stats.csv")
            current_all_merged_df.to_csv(f"{current_merged_fp}{cc}_All_Subreddits_{frequencies[f]}_Aggregated_Stats.csv")

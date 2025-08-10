import pandas as pd
import os

base_path = "All_Crypto_Data/Crypto_Sub_Reddit_Data/Raw_Sub_Reddit_Text_Data/"

for cc in os.listdir(base_path):
    current_cc_path = f'{base_path}{cc}/Extracted_Reddit_Data/'

    for csv in os.listdir(current_cc_path):
        current_df = pd.read_csv(f'{current_cc_path}{csv}')

        if csv.endswith('submissions.csv'):
            current_df = current_df.rename(columns={'author':'POST_AUTHOR',
                                                    'title':'POST_TITLE',
                                                    'score':'POST_SCORE',
                                                    'created':'TIMESTAMP',
                                                    'link':'POST_LINK',
                                                    'text':'POST_TEXT',
                                                    'url':'POST_URL',})
            
            current_df = current_df[['TIMESTAMP', 'POST_AUTHOR', 'POST_TITLE', 'POST_TEXT', 'POST_SCORE', 'POST_LINK', 'POST_URL']]
        else:
            current_df = current_df.rename(columns={'author':'COMMENT_AUTHOR',
                                                    'score':'COMMENT_SCORE',
                                                    'created':'TIMESTAMP',
                                                    'link':'COMMENT_LINK',
                                                    'body':'COMMENT_TEXT',})
            
            current_df = current_df[['TIMESTAMP', 'COMMENT_AUTHOR', 'COMMENT_TEXT', 'COMMENT_SCORE', 'COMMENT_LINK']]

        current_df.to_csv(f'{current_cc_path}{csv}', index=False)

import re
import regex
import os
import html
import contractions
import pandas as pd
from datetime import datetime
from nltk.corpus import words


def prepare_text(text, 
                 keep_digits=False, 
                 keep_emojis=False):
    text = html.unescape(text) # Handle Encoded Characters
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text) # Replace Links of Type [Text](URL) with Text based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = re.sub(r'\[([^\]]+)\]\[[^)]+\]', r'\1', text) # Replace Links of Type [Text][URL] with Text based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = re.sub(r'<([^>]+)>', r'\1', text) # Replace Links of Type <URL> with URL based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = re.sub(r'\[[^\]]+\]:', '', text) # Remove Portion of Alternative Link based on on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = re.sub(r'\b(?:https?://|www\.|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(/[^\s]*)?\b', '', text) # Replace URLs with Empty String

    text = re.sub(r'\br/([\w\d_]+)', r'\1', text) # Handle References to other Reddits based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = re.sub(r'\bu/([\w\d_]+)', r'\1', text) # Handle References to other Users based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide

    text = re.sub(r'\*(.*?)\*', r'\1', text) # Handle Italic Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = re.sub(r'\_(.*?)\_', r'\1', text) # Handle Italic Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide

    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) # Handle Bold Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = re.sub(r'\_\_(.*?)\_\_', r'\1', text) # Handle Bold Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide

    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text) # Handle Bold-Italic Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = re.sub(r'\_\_\_(.*?)\_\_\_', r'\1', text) # Handle Bold-Italic Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide

    text = re.sub(r'\~\~(.*?)\~\~', r'\1', text) # Handle Strikethrough Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = re.sub(r'\>\!(.*?)\!\<', r'\1', text) # Handle Spolier Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide

    text = re.sub(r'\^(.*?)', r'\1', text) # Handle Superscript Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = re.sub(r'\^\((.*?)\)', r'\1', text) # Handle Superscript Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide

    text = re.sub(r'\`(.*?)\`', r'\1', text) # Handle Code Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = re.sub(r'~~~.*?~~~', '', text, flags=re.DOTALL) # Handle Code Blocks Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL) # Handle Code Blocks Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide

    text = re.sub(r'^\s*#{1,6}\s*', '', text) # Handle Headings Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = text.replace('===', '') # Handle Headings Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = text.replace('---', '') # Handle Headings Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = text.replace('___', '') # Handle Headings Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide

    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE) # Handle Lists Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = re.sub(r'^\s*\d+\)\s+', '', text, flags=re.MULTILINE) # Handle Lists Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE) # Handle Lists Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide

    text = re.sub(r'(?:\|.*\|\n)+', '', text) # Handle Tables Markdown based on https://support.reddithelp.com/hc/en-us/articles/360043033952-Formatting-Guide

    text = re.sub(r'\b(?:bc1|[13])[a-zA-Z0-9]{25,39}\b', '', text) # Replace BTC Wallet Addresses with Empty String
    text = re.sub(r'\b0x[a-fA-F0-9]{40}\b', '', text) # Replace ETH Wallet Addresses with Empty String
    text = re.sub(r'\b[48][0-9AB][1-9A-HJ-NP-Za-km-z]{93,105}\b', '', text) # Replace XRP Wallet Addresses with Empty String
    text = re.sub(r'\br[0-9a-zA-Z]{24,34}\b', '', text) # Replace XMR Wallet Addresses with Empty String
    text = re.sub(r'\b[LM3][a-km-zA-HJ-NP-Z1-9]{25,34}\b', '', text) # Replace LTC Wallet Addresses with Empty String
    text = re.sub(r'\bltc1[0-9a-z]{39,59}\b', '', text) # Replace LTC Wallet Addresses with Empty String

    text = text.replace('\n', '. ').replace('\t', '. ') # Replace Enters and Tabs with spaces
    text = text.replace('_', '') # Replace Underscores with Empty String
    text = re.sub(r'\s&\s', ' and ', text) # Replace & with and
    text = re.sub(r'&(?!\s)|(?<!\s)&', ' and ', text) # Replace & with and

    try:
        text = contractions.fix(text) # Expand Contractions
    except IndexError:
        text = text 

    if keep_digits and keep_emojis:
        text = re.sub(re.compile(r'[^\w\s\-+!?.,%\U0001F600-\U0001F64FF\U0001F680-\U0001F6FF\U0001F300-\U0001F5FF]', flags=re.UNICODE), '', text)
    elif keep_emojis:
        text = re.sub(r'(?<!\w)[+-]?\s?\d+[.,]?\d*(?=\s?%?)', '', text)
        text = regex.sub(regex.compile(r'[^\p{L}\s\-!?.,\U0001F600-\U0001F64FF\U0001F680-\U0001F6FF\U0001F300-\U0001F5FF]', flags=re.UNICODE), '', text)
    elif keep_digits:
        text = re.sub(re.compile(r'[^\w\s\-+!?.,%]'), '', text)
    else:
        text = re.sub(r'(?<!\w)[+-]?\s?\d+[.,]?\d*(?=\s?%?)', '', text)
        text = regex.sub(regex.compile(r'[^\p{L}\s\-!?.,]'), '', text)
    
    text = re.sub(r'\-+', '-', text) # Replace multiple successive -'s
    text = re.sub(r'\++', '+', text) # Replace multiple successive +'s
    text = re.sub(r'\,+', ',', text) # Replace multiple successive ,'s
    text = re.sub(r'\%+', '%', text) # Replace multiple successive %'s
    text = re.sub(r'\s+', ' ', text).strip() # Replace Excessive Spaces
    
    text = re.sub(r'\s+(?=\.)', '', text) # Remove space in front of .
    text = re.sub(r'\s+(?=,)', '', text) # Remove space in front of ,
    text = re.sub(r'\s+(?=\!)', '', text) # Remove space in front of !
    text = re.sub(r'\s+(?=\?)', '', text) # Remove space in front of ?
    
    text = re.sub(r'(?<!\w)\.', '', text) # Remove . if not preceeded by letters / digits
    text = re.sub(r'(?<!\w),', '', text) # Remove , if not preceeded by letters / digits
    text = re.sub(r'(?<!\w)\%', '', text) # Remove % if not preceeded by letters / digits
    text = re.sub(r'\+(?!\w)', '', text) # Remove + if not followed by letters / digits
    text = re.sub(r'-(?!\w)', '', text) # Remove - if not followed by letters / digits
 
    return text.lower()


def bot_removal(df, column_name='COMMENT_AUTHOR'):
    unique_authors = list(df[column_name].unique())
    authors_to_filter_out = []
    bot_words = [w for w in words.words() if 
                 ('bot' in w.lower()) and 
                 ('bot' != w.lower()) and 
                 ('bota' != w.lower()) and 
                 ('bote' != w.lower()) and 
                 ('bott' != w.lower()) and 
                 ('cabot' != w.lower()) and
                 ('botch' != w.lower()) and
                 ('robot' != w.lower())]

    for a in unique_authors:
        if 'bot' in a.lower():
            substring_count = 0

            for bw in bot_words:
                if bw.lower() in a.lower():
                    substring_count+=1

            if not substring_count:
                authors_to_filter_out.append(a)
        elif ('auto' in a.lower()) or ('mod' in a.lower()):
            authors_to_filter_out.append(a)
        else:
            continue

    df = df[~df[column_name].isin(authors_to_filter_out)].reset_index(drop=True)

    return df


def remove_deleted_removed_nan_comments(comments_df):
    deletion_indicators = ['Removed by Reddit', 
                           'Removed By AutoModerator', 
                           'deleted by user', 
                           '[removed]', 
                           '[deleted]', 
                           '[ removed ]', 
                           '[ deleted ]', 
                           'Removed By Mods',
                           'Removed By Moderator']

    comments_df = comments_df.dropna(subset=['COMMENT_TEXT']).reset_index(drop=True)
    comments_df = comments_df[comments_df['COMMENT_TEXT'].str.len() > 0].reset_index(drop=True)

    comments_df = comments_df[~comments_df['COMMENT_TEXT'].str.lower().isin(['removed', 'deleted'])].reset_index(drop=True)
    comments_df = comments_df[~comments_df['COMMENT_TEXT'].str.contains('|'.join(re.escape(s) for s in deletion_indicators), case=False)].reset_index(drop=True)

    return comments_df


def remove_deleted_or_removed_posts(posts_df):
    deletion_indicators = ['Removed by Reddit', 
                           'Removed By AutoModerator', 
                           'deleted by user', 
                           '[removed]', 
                           '[deleted]', 
                           '[ removed ]', 
                           '[ deleted ]', 
                           'Removed By Mods',
                           'Removed By Moderator']

    posts_df = posts_df.dropna(subset=['POST_TITLE']).reset_index(drop=True)
    posts_df = posts_df[~posts_df['POST_TITLE'].str.lower().isin(['removed', 'deleted'])].reset_index(drop=True)
    posts_df = posts_df[~posts_df['POST_TITLE'].str.contains('|'.join(re.escape(s) for s in deletion_indicators), case=False)].reset_index(drop=True)

    posts_df['POST_TEXT'] = posts_df['POST_TEXT'].fillna('').reset_index(drop=True)
    posts_df['POST_TEXT'] = posts_df['POST_TEXT'].str.replace(r'\b(removed|deleted)\b', '', case=False, regex=True)
    posts_df['POST_TEXT'] = posts_df['POST_TEXT'].apply(lambda x: '' if re.search('|'.join(re.escape(s) for s in deletion_indicators), x, flags=re.IGNORECASE) else x)
    
    return posts_df


def preprocessing_pipeline_comments(df,
                                    launch_date, 
                                    keep_digits=False, 
                                    keep_emojis=False):
    original_c_length = len(df)

    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], 
                                     format='%H:%M %d-%m-%Y')
    df = df[df['TIMESTAMP'] >= launch_date]
    df = remove_deleted_removed_nan_comments(df)
    df = bot_removal(df)
    df['COMMENT_TEXT'] = df['COMMENT_TEXT'].apply(lambda x: prepare_text(x, 
                                                                         keep_digits, 
                                                                         keep_emojis))
    df = df[df['COMMENT_TEXT'].str.contains(r'[a-zA-Z]', 
                                            na=False)].reset_index(drop=True)
    df = df[df['COMMENT_TEXT'].str.split().str.len() >= 4].reset_index(drop=True)
    df = df[~df['COMMENT_TEXT'].str.contains('|'.join(re.escape(s) for s in ['giveaway', 'airdrop', 'free', 'launch', 'presale', 'early access', 'follow me', 'check bio', 'comment for', 'nsfw']), 
                                             na=False, 
                                             case=False)].reset_index(drop=True)
    df = df[~df['COMMENT_TEXT'].str.match(r'[^a-zA-Z0-9\s\-+!?.,%\U0001F600-\U0001F64FF\U0001F680-\U0001F6FF\U0001F300-\U0001F5FF]', 
                                          flags=re.UNICODE,
                                          na=False)].reset_index(drop=True)
    df = df.drop_duplicates(subset='COMMENT_TEXT', 
                            keep=False, 
                            ignore_index=True)
    new_c_length = len(df)

    print(f"\t\t\t\t- Original Comments DF Length: {original_c_length}, New Comments DF Length: {new_c_length}")
    print(f"\t\t\t\t- Filtered Out Comments DF Row Count: {original_c_length - new_c_length}")

    return df[['TIMESTAMP', 'COMMENT_TEXT', 'COMMENT_SCORE', 'POST_ID']]


def preprocessing_pipeline_posts(df, 
                                 launch_date,
                                 keep_digits=False, 
                                 keep_emojis=False):
    original_sm_length = len(df)
    
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'],
                                     format='%H:%M %d-%m-%Y')
    df = df[df['TIMESTAMP'] >= launch_date]
    df = remove_deleted_or_removed_posts(df)
    df = bot_removal(df, 
                     column_name='POST_AUTHOR')
    df['POST_TEXT'] = df.apply(lambda x: f'{x.POST_TITLE}. {x.POST_TEXT}', 
                               axis=1)
    df['POST_TEXT'] = df['POST_TEXT'].apply(lambda x: prepare_text(x, 
                                                                   keep_digits, 
                                                                   keep_emojis))
    df = df[df['POST_TEXT'].str.contains(r'[a-zA-Z]', 
                                         na=False)].reset_index(drop=True)
    df = df[df['POST_TEXT'].str.split().str.len() >= 4].reset_index(drop=True)
    df = df[~df['POST_TEXT'].str.contains('|'.join(re.escape(s) for s in ['giveaway', 'airdrop', 'free', 'launch', 'presale', 'early access', 'follow me', 'check bio', 'comment for', 'nsfw']), 
                                          na=False, 
                                          case=False)].reset_index(drop=True)
    df = df[~df['POST_TEXT'].str.match(r'[^a-zA-Z0-9\s\-+!?.,%\U0001F600-\U0001F64FF\U0001F680-\U0001F6FF\U0001F300-\U0001F5FF]', 
                                      flags=re.UNICODE,
                                      na=False)].reset_index(drop=True)
    df = df.drop_duplicates(subset='POST_TEXT', 
                            keep=False, 
                            ignore_index=True)

    new_sm_length = len(df)

    print(f"\t\t\t\t- Original Posts DF Length: {original_sm_length}, New Posts DF Length: {new_sm_length}")
    print(f"\t\t\t\t- Filtered Out Posts DF Row Count: {original_sm_length - new_sm_length}")

    return df[['TIMESTAMP', 'POST_TEXT', 'POST_SCORE', 'POST_ID']]


base_path = 'All_Crypto_Data/Crypto_Sub_Reddit_Data/Raw_Sub_Reddit_Text_Data/'
preprocessed_base_path = 'All_Crypto_Data/Crypto_Sub_Reddit_Data/Preprocessed_Sub_Reddit_Text_Data/'

cc_mapping = {"BTC": "Bitcoin",
              "ETH": "Ethereum",
              "LTC": "Litecoin",
              "XRP": "Ripple",
              "XMR": "Monero"}

launch_date_mapping = {"BTC": datetime(2009, 1, 3),
                       "ETH": datetime(2015, 7, 30),
                       "LTC": datetime(2011, 10, 13),
                       "XRP": datetime(2012, 1, 1),
                       "XMR": datetime(2014, 4, 18)}

preprocessing_level_mapping = {"Fully_Preprocessed_Sub_Reddit_Text_Data/": [False, False],
                               "Preprocessed_Sub_Reddit_Text_Data_with_Digits/": [True, False],
                               "Preprocessed_Sub_Reddit_Text_Data_with_Emojis/": [False, True],
                               "Preprocessed_Sub_Reddit_Text_Data_with_Digits_and_Emojis/": [True, True],}


for cc in reversed(cc_mapping.keys()):
    subreddits = sorted(list(set([re.sub(r'_(submissions|comments)\.csv$', '', sr)
                                  for sr in os.listdir(f'{base_path}{cc}/Extracted_Reddit_Data/')])))
    crypto_subreddits = sorted(list(set([re.sub(r'_(submissions|comments)\.csv$', '', csr)
                                         for csr in os.listdir(f'{base_path}Crypto/Extracted_Reddit_Data/')])))
    
    print(f"Current Crypto: {cc}:")
    print(f"\t- {cc} Subreddits:")

    for sr in subreddits:
        print(f"\t\t- Current Subreddit: {sr}")

        current_sr_posts = pd.read_csv(f'{base_path}{cc}/Extracted_Reddit_Data/{sr}_submissions.csv') 
        current_sr_comments = pd.read_csv(f'{base_path}{cc}/Extracted_Reddit_Data/{sr}_comments.csv')

        current_sr_posts['POST_ID'] = current_sr_posts['POST_LINK'].apply(lambda x: f'/{x.split("/comments/")[1].split("/")[0]}/')
        current_sr_comments['POST_ID'] = current_sr_comments['COMMENT_LINK'].apply(lambda x: f'/{x.split("/comments/")[1].split("/")[0]}/')

        for pl in preprocessing_level_mapping.keys():
            current_preprocessed_base_path = f'{preprocessed_base_path}{pl}{cc}/'
            current_preprocessed_sr_posts_path = f'{current_preprocessed_base_path}{sr}_posts.csv'
            current_preprocessed_sr_comments_path = f'{current_preprocessed_base_path}{sr}_comments.csv'

            print(f"\t\t\t- Current Preprocessing Level: {pl[:-1]}")
            print(f"\t\t\t\t- Current Preprocessing Path: {current_preprocessed_base_path}")

            if not os.path.exists(current_preprocessed_base_path):
                os.makedirs(current_preprocessed_base_path)

            if not os.path.exists(current_preprocessed_sr_posts_path):
                current_preprocessed_sr_posts = preprocessing_pipeline_posts(df=current_sr_posts, 
                                                                            launch_date=launch_date_mapping[cc],
                                                                            keep_digits=preprocessing_level_mapping[pl][0],
                                                                            keep_emojis=preprocessing_level_mapping[pl][1])
                current_preprocessed_sr_posts.to_csv(current_preprocessed_sr_posts_path, 
                                                     index=False)
            
            if not os.path.exists(current_preprocessed_sr_comments_path):
                current_preprocessed_sr_comments = preprocessing_pipeline_comments(df=current_sr_comments, 
                                                                                launch_date=launch_date_mapping[cc],
                                                                                keep_digits=preprocessing_level_mapping[pl][0],
                                                                                keep_emojis=preprocessing_level_mapping[pl][1])
                current_preprocessed_sr_comments.to_csv(current_preprocessed_sr_comments_path, 
                                                        index=False)

    
    print(f"\t- Crypto Subreddits:")


    for csr in crypto_subreddits:
        print(f"\t\t- Current Subreddit: {csr}")

        current_csr_posts = pd.read_csv(f'{base_path}Crypto/Extracted_Reddit_Data/{csr}_submissions.csv') 
        current_csr_comments = pd.read_csv(f'{base_path}Crypto/Extracted_Reddit_Data/{csr}_comments.csv')

        current_csr_posts['POST_ID'] = current_csr_posts['POST_LINK'].apply(lambda x: f'/{x.split("/comments/")[1].split("/")[0]}/')
        current_csr_comments['POST_ID'] = current_csr_comments['COMMENT_LINK'].apply(lambda x: f'/{x.split("/comments/")[1].split("/")[0]}/')

        current_relevant_csr_posts = current_csr_posts[current_csr_posts['POST_TEXT'].str.contains(fr'\b({cc}|{re.escape("$" + cc)}|{cc_mapping[cc]}|{re.escape("$" + cc_mapping[cc])})\b', 
                                                                                                   case=False, 
                                                                                                   na=False) |
                                                       current_csr_posts['POST_TITLE'].str.contains(fr'\b({cc}|{re.escape("$" + cc)}|{cc_mapping[cc]}|{re.escape("$" + cc_mapping[cc])})\b', 
                                                                                                    case=False, 
                                                                                                    na=False)].reset_index(drop=True)
        current_relevant_csr_comments = current_csr_comments[current_csr_comments['COMMENT_TEXT'].str.contains(fr'\b({cc}|{re.escape("$" + cc)}|{cc_mapping[cc]}|{re.escape("$" + cc_mapping[cc])})\b', 
                                                                                                               case=False, 
                                                                                                               na=False) |
                                                             current_csr_comments['POST_ID'].isin(list(current_relevant_csr_posts['POST_ID']))].reset_index(drop=True)
        

        for pl in preprocessing_level_mapping.keys():
            current_preprocessed_base_path = f'{preprocessed_base_path}{pl}{cc}/'
            current_preprocessed_csr_posts_path = f'{current_preprocessed_base_path}{cc}_In_{csr}_posts.csv'
            current_preprocessed_csr_comments_path = f'{current_preprocessed_base_path}{cc}_In_{csr}_comments.csv'

            print(f"\t\t\t- Current Preprocessing Level: {pl[:-1]}")
            print(f"\t\t\t\t- Current Preprocessing Path: {current_preprocessed_base_path}")

            if not os.path.exists(current_preprocessed_base_path):
                os.makedirs(current_preprocessed_base_path)
            
            if not os.path.exists(current_preprocessed_csr_posts_path):
                current_preprocessed_csr_posts = preprocessing_pipeline_posts(df=current_relevant_csr_posts, 
                                                                              launch_date=launch_date_mapping[cc],
                                                                              keep_digits=preprocessing_level_mapping[pl][0],
                                                                              keep_emojis=preprocessing_level_mapping[pl][1])
                current_preprocessed_csr_posts.to_csv(current_preprocessed_csr_posts_path, 
                                                      index=False)
            
            if not os.path.exists(current_preprocessed_csr_comments_path):
                current_preprocessed_csr_comments = preprocessing_pipeline_comments(df=current_relevant_csr_comments, 
                                                                                    launch_date=launch_date_mapping[cc],
                                                                                    keep_digits=preprocessing_level_mapping[pl][0],
                                                                                    keep_emojis=preprocessing_level_mapping[pl][1])
                current_preprocessed_csr_comments.to_csv(current_preprocessed_csr_comments_path, 
                                                         index=False)

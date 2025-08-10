import time
import os
import pandas as pd 
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                           num_labels = 3)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=128, truncation=True, padding = 'max_length')

selected_cryptos = ['BTC', 'ETH', 'LTC', 'XMR', 'XRP']
preprocessed_sub_reddit_text_data_base_path = "All_Crypto_Data/Crypto_Sub_Reddit_Data/Preprocessed_Sub_Reddit_Text_Data/"

for preprocess_type in os.listdir(preprocessed_sub_reddit_text_data_base_path):
    for cc in os.listdir(f"{preprocessed_sub_reddit_text_data_base_path}{preprocess_type}/"):
        for text_csv in os.listdir(f"{preprocessed_sub_reddit_text_data_base_path}{preprocess_type}/{cc}/"):
            if "WallStreetBetsCrypto" in text_csv:
                current_text_df = pd.read_csv(f"{preprocessed_sub_reddit_text_data_base_path}{preprocess_type}/{cc}/{text_csv}")

                if 'posts' in text_csv:
                    texts = list(current_text_df['POST_TEXT'])[:50]
                else:
                    texts = list(current_text_df['COMMENT_TEXT'])[:50]
                
                print(pipe(texts))


# post_1 & post_3 = bullish, post_2 = bearish
# post_1 = " see y'all tomorrow and can't wait to see ada in the morning, i wonder what price it is going to be at. 😎🐂🤠💯😴, bitcoin is looking good go for it and flash by that 45k. "
# post_2 = "  alright racers, it’s a race to the bottom! good luck today and remember there are no losers (minus those who invested in currency nobody really uses) take your marks... are you ready? go!!" 
# post_3 = " i'm never selling. the whole market can bottom out. i'll continue to hold this dumpster fire until the day i die if i need to." 
# df_posts = [post_1, post_2, post_3]



# start_1 = time.time()

# for post in df_posts:
#     print(pipe(post))

# end_1 = time.time()

# start_2 = time.time()

# print(pipe(df_posts))

# end_2 = time.time()

# print(f"Time 1: {end_1 - start_1}")
# print(f"Time 2: {end_2 - start_2}")

# preds = pipe(df_posts)
# print(preds)






# print(outputs)

# import torch
# from transformers import TextClassificationPipeline, AutoModel, AutoTokenizer

# from transformers import BertTokenizer, BertModel
# import torch

# tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert", 
#                                           use_fast=True)
# model = AutoModel.from_pretrained("ElKulako/cryptobert", 
#                                   num_labels=3)

# text = "see y'all tomorrow and can't wait to see ada in the morning, i wonder what price it is going to be at. 😎🐂🤠💯😴, bitcoin is looking good go for it and flash by that 45k."
# inputs = tokenizer(text, return_tensors='pt')

# with torch.no_grad():
#     outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state
# print("Shape of the last hidden state (embeddings):", last_hidden_states.shape)

# tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
# for token, embedding in zip(tokens, last_hidden_states[0]):
#     print(f"Token: {token}, Embedding Dimension: {embedding.shape}, Embedding (first 5 components): {embedding[:5]}...")  # Display first 5 components for brevity


# # Load pre-trained model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# # Define the text
# text = "The quick brown fox jumps over the lazy dog."

# # Tokenize the text
# inputs = tokenizer(text, return_tensors='pt')

# # Obtain the embeddings
# with torch.no_grad():
#     outputs = model(**inputs)

# # Extract the last hidden state (embeddings)
# last_hidden_states = outputs.last_hidden_state

# # Print the dimensions of the embeddings
# print("Shape of the last hidden state (embeddings):", last_hidden_states.shape)

# # Print embeddings for each token along with their vector dimension
# tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
# for token, embedding in zip(tokens, last_hidden_states[0]):
#     print(f"Token: {token}, Embedding Dimension: {embedding.shape}, Embedding (first 5 components): {embedding[:5]}...")  # Display first 5 components for brevity

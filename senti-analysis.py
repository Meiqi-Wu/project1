#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:15:12 2019

@author: wumeiqi
"""

#%% Import the packages
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from nltk import word_tokenize
from tqdm import tqdm # Instantly make loops show a smart progress meter
#%% prepare the data
# Extract reviews and ratings of each restaurant and create a dataframe with it
ratings = []
for name, reviews in tqdm(zip(df['name'], df['reviews_list'])):
    for score, doc in reviews:
        score = score.strip('Rated').strip()
        doc = doc.strip('RATED').strip()
        score = float(score)
        ratings.append([name, score, doc])
del name, reviews, score, doc

ratings_df = pd.DataFrame(ratings, columns=['name', 'score', 'review'])
ratings_df.head()
# clear each review text
ratings_df['review'] = ratings_df['review'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))
#ratings_df.to_csv('./DATA/Ratings.csv')
#%%
# divide reviews as negative and positive based on scores
ratings_df['senti'] = ratings_df['score'].apply(lambda x: 1 if int(x)>2.5 else 0)
# tokenize the data and vectorize the reviews
max_features = 1000
tokenizer = Tokenizer(num_words = max_features, split=' ')
tokenizer.fit_on_texts(ratings_df['review'].values)
X = tokenizer.texts_to_sequences(ratings_df['review'].values)
X = pad_sequences(X)

#%% Build the model



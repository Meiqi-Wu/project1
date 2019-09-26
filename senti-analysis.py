#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this script, we conduct sentiment analysis of reviews.

Created on Mon Sep 18 21:15:12 2019

@author: wumeiqi
"""

#%% Import the packages
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from nltk import word_tokenize
from tqdm import tqdm # Instantly make loops show a smart progress meter
import re
import pandas as pd
from sklearn.model_selection import train_test_split
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
max_features = 1000 # The maximu number of words to keep
# create a tokenizer
tokenizer = Tokenizer(num_words = max_features, split=' ')
# fit the tokenizer on the review text
tokenizer.fit_on_texts(ratings_df['review'].values)
X = tokenizer.texts_to_sequences(ratings_df['review'].values)
X = pad_sequences(X)

#%% Build the model
embed_dim = 32
lstm_out = 32

model = Sequential()
# input_dim: len(o), output_dim=len(e), E=len(e)*len(o)
model.add(Embedding(input_dim=max_features, output_dim=embed_dim, input_length=X.shape[1]))
model.add(LSTM(units=lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#%% Train the model
# convert senti(0/1) to one-hot vectors
Y = pd.get_dummies(ratings_df['senti'].astype(int)).values
# split the training and test set
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3)
print(Xtrain.shape, Ytrain.shape)
print(Xtest.shape, Ytest.shape)

#%% 
batch_size=1600
model.fit(Xtrain, Ytrain, epochs=5, batch_size=batch_size)


#%% Evaluation
loss, acc = model.evaluate(Xtest, Ytest, batch_size=batch_size)
print('Test set loss: {}'.format(loss))
print('Test set accuracy: {}'.format(acc))










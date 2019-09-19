#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 22:36:48 2019

@author: wumeiqi
"""

#%% Importing Libraries
import pandas as pd
import numpy as np
import ast


#%% Import the data
df = pd.read_csv('../DATA/zomato.csv', nrows=1000)
print('dataset contains {} rows and {} columns'.format(df.shape[0],df.shape[1]))
df.info()
df.head()
df.dtypes


#%% Data Cleaning

# Removing unnecessary columns
column_to_drop=['url', 'phone','location']
df.drop(columns=column_to_drop, axis=1, inplace=True)

# Renaming columns
df.rename(columns={'approx_cost(for two people)':'average_cost',
                   'listed_in(type)':'type','listed_in(city)':'city'},inplace=True)

# Adjust the column type
df.oneline_order.replace(('Yes', 'No'), (True, False), inplace=True)
df.book_table.replace(('Yes', 'No'), (True, False), inplace=True)
df.average_cost = df.average_cost.apply(lambda x: int(x.replace(',', '')))
# update the 'review_list' from string to list of tuples
df.reviews_list = df.reviews_list.apply(lambda x: ast.literal_eval(x))

# Removing duplicates
df.drop_duplicates(inplace=True)

# Checking NaN and Null
print("Percentage of null and na values in df")
((df.isnull() | df.isna()).sum()*100/ df.index.size).round(2)

#%% Check the 'rate' column
df.rate.isna().sum()
df.rate.unique()

# convert the revert "New" & "-" to np.nan
df.rate.replace(('NEW','-'), np.nan, inplace=True)
df.rate = df.rate.astype('str')
df.rate = df.rate.apply(lambda x: x.replace('/5', '').strip())
df.rate = df.rate.astype('float')

# 'review_list' column : generate rate from review
def get_rate(x):
    '''
    get the value of rate from a list of reviews
    :type x : list
    :rtype: float
    '''
    if not x or len(x) <= 1: return None 
    rate = [float(i[0].replace('Rated','').strip()) for i in x if type(i[0]==str)]
    return round(sum(rate)/len(rate), 1)
    
# create a new column
df['review_rate'] = df.reviews_list.apply(lambda x: get_rate(x))
df.loc[:,['rate', 'review_rate']].sample(10, random_state=1)

# Update the 'rate' column 
nan_index = df.query('rate != rate & review_rate == review_rate').index
for i in nan_index:
    df.loc[i, 'rate']=df.loc[i, 'review_rate']
# drop the review_rate column
df.drop(columns='review_rate', axis=1, inplace=True)


#%% Drop null
df.dropna(subset=['rate', 'average_cost'], inplace=True)

print("Percentage of null and na values in df")
((df.isnull() | df.isna()).sum()*100/ df.index.size).round(2)


#%% Clean the 'dish_liked' column
# make lower case
df.dish_liked = df.dish_liked.apply(lambda x: x.lower().strip() if isinstance(x, str) else x)

# collect the dishes' name and make a menu list for all kind of dishes
temp = [i.split(',') for i in df.dish_liked.tolist() if isinstance(i, str)]
menu_list = [e.strip() for i in temp for e in i]
menu_set = set(menu_list)
del temp

# clean the 'reviews_list' text
def clear_text(x):
    return ' '.join([i[1].replace("RATED\n  ",'') for i in x]).encode('utf8').decode('ascii',errors='replace').\
           replace('?','').replace('�','').replace('\n','').replace('.',' ').strip().lower()
df['review_text']=df.reviews_list.apply(lambda x: clear_text(x))

# create a new column for the dishes reviewed in review_text
df['dish_reviewed']=df.review_text.apply(lambda x: ', '.join(list(menu_set.intersection(x.split(' ')))))
df.query('dish_liked!=dish_liked')[['dish_liked', 'dish_reviewed']].sample(5, random_state=1)

nan_index = df.query('dish_liked!=dish_liked & dish_reviewed == dish_reviewed').index
for i in nan_index:
    df['dish_liked'][i]=df['dish_reviewed'][i]

del menu_list
del menu_set
print("Percentage of null and na values in df")
((df.isnull() | df.isna()).sum()*100/ df.index.size).round(2)

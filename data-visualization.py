#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:13:44 2019

@author: wumeiqi
"""

import seaborn as sns
import matplotlib.pyplot as plt


#%% Restaurant location (city) distribution plot
plt.rcParams['figure.figsize']=8, 6
g = sns.countplot(x='city', data=df)
g.set_xticklabels(g.get_xticklabels(), rotation=90, ha='right')
fig=g.get_figure()
plt.title('Location distribution', fontsize=20)
plt.xlabel('City', fontsize=16)
plt.ylabel('Count', fontsize=16)
fig.savefig('City_distribution.png')

#%% Restaurant type distribution plot
plt.rcParams['figure.figsize']=20, 6
g = sns.countplot(x='rest_type', data=df)
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha='right', fontsize=8)
fig=g.get_figure()
plt.title('Restaurant type distribution', fontsize=20)
plt.xlabel('Type', fontsize=16)
plt.ylabel('Count', fontsize=16)
fig.savefig('Rest_type_distribution.png')

#%% Pie chart of the 6 most popular restaurant types
plt.rcParams['figure.figsize']=8, 6
rest_type_count = df['rest_type'].value_counts().sort_values(ascending=False)
slices=[rest_type_count[0], rest_type_count[1], rest_type_count[2],
        rest_type_count[3], rest_type_count[4], rest_type_count[5]]
labels=['Quick Bites', 'Casual Dining', 'Cafe', 'Dessert Parlor', 'Delivery',
        'Takeaway, Delivery']
colors=['Red', 'Orange', 'Yellow','Green','Blue','purple']
plt.pie(slices, colors=colors, labels=labels, shadow=True,autopct='%1.0f%%')
plt.title('Percentage of the 6 most popular restaurante types', 
          bbox={'facecolor':'2', 'pad':5})
plt.savefig('Pie_rest_type.png')

#%% Rating distribution
plt.rcParams['figure.figsize']=8, 6
rating=df['rate'].dropna()
g = sns.distplot(rating, bins=30)
fig = g.get_figure()
plt.xlabel('Rate', fontsize=16)
plt.title('Rate distribution', fontsize=20)
fig.savefig('Rate_distribution.png')

#%% Relation between rate the book_table
plt.rcParams['figure.figsize']=10, 6
x = pd.crosstab(df['rate'], df['book_table'], normalize='index')
x.plot.bar(stacked=True)
plt.legend(loc="upper right")
plt.title('Rate v.s. Book table', fontsize=20)
plt.xlabel('Rate', fontsize=16)
plt.savefig('Rate_booktable.png')

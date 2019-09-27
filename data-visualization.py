#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:13:44 2019

@author: wumeiqi
"""

import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=8, 6
#%%
# Restaurant location (city) distribution plot
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

#%% Pie chart of restaurant types
rest_type_count = df['rest_type'].value_counts().sort_values(ascending=True)


